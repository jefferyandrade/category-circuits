"""Truthful FD-geometry provenance for a device-resident sampled-D endpoint.

The historical literal-J rank-replay receipt proves a host-custody replay and
therefore must retain its nonzero file-read/H2D/D2H telemetry.  A live
device-resident D session reaches the same scientific FD coordinate by a
different path: the registered A/J capture is accumulated directly on device
and exactly one selected-rank portable snapshot is emitted at the endpoint.

This module keeps those provenance classes disjoint.  Its normalizer exposes
only the common scientific coordinate used by downstream uncertainty code;
it never converts a device source into a host-replay receipt and grants no D,
C, K, O, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Final, Mapping, NoReturn

from .contracts import content_hash
from .modern_v1_sampled_composition_backend import (
    V1SampledCompositionRegisteredDPrefixBinding,
    V1SampledCompositionRegisteredDTransaction,
    build_v1_sampled_composition_registered_d_prefix_binding,
)
from .modern_v1_sampled_device_resident_d_session import (
    DEVICE_RESIDENT_D_REGISTERED_RANK_MATERIALIZATION_MODE,
    V1SampledDeviceResidentDMaterializationRankReceipt,
    V1SampledDeviceResidentDSessionMaterialization,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowLivePrefixAttachment,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJFDRankReplayReceipt,
    V1SampledLiteralJOriginalCaptureBinding,
)
from .modern_v1_proposal_statistics import (
    V1StratifiedCorrectedAttributionMoments,
)
from .modern_v1_fqr_adaptive_rank import V1FQREndogenousRankDPrefixV2
from .modern_v1_stratified_fd_gpu import V1StratifiedFDDesign


SAMPLED_DEVICE_FINAL_RANK_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_device_final_rank_source_v1"
)
SAMPLED_DEVICE_FINAL_RANK_SOURCE_AUTHORITY: Final = (
    "device_resident_final_rank_scientific_source_only_no_host_replay_"
    "rank_stability_D_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_FD_GEOMETRY_SOURCE_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_sampled_FD_geometry_source_coordinate_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-device-final-rank-source-"
_RECEIPT_PREFIX: Final = _PREFIX + "receipt-"
_COORDINATE_PREFIX: Final = _PREFIX + "common-coordinate-"
_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-context-axis-"
_DRAW_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-draw-axis-"
_STRATUM_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-stratum-axis-"
_PACKET_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-packet-axis-"
_SOURCE_MEASUREMENT_AXIS_PREFIX: Final = (
    "partition-free-v1-literal-j-replay-source-measurement-axis-"
)
_ISSUER_TOKEN: Final = object()


class V1SampledFDGeometrySourceProvenance(str, Enum):
    HOST_CUSTODY_RANK_REPLAY = "host_custody_rank_replay"
    DEVICE_RESIDENT_FINAL_RANK = "device_resident_final_rank"


class V1SampledDeviceFinalRankSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_MISMATCH = "source_mismatch"
    MIXED_PROVENANCE = "mixed_provenance"
    STALE_SOURCE = "stale_source"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledDeviceFinalRankSourceRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledDeviceFinalRankSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledDeviceFinalRankSourceRefusalCode:
            raise TypeError("device-final-rank refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledDeviceFinalRankSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledDeviceFinalRankSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _flatten(axis: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    return tuple(item for packet in axis for item in packet)


@dataclass(frozen=True, slots=True)
class V1SampledDeviceFinalRankSourceReceipt:
    original_capture_binding_hash: str
    device_materialization_hash: str
    materialization_rank_receipt_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    rank_invariant_raw_prefix_hash: str
    literal_j_source_receipt_hash: str
    row_prefix_manifest_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    corrected_moments_state_hash: str
    requested_physical_rank: int
    registered_look_count: int
    packet_count: int
    context_count: int
    original_capture_batch_count: int
    original_vjp_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    packet_axis_hash: str
    source_measurement_axis_hash: str
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    literal_j_stream_identity_hash: str
    corrected_a_stream_identity_hash: str
    device_identity: str
    final_route_scale_d2h_bytes: int
    host_custody_packet_read_count: int = 0
    host_custody_payload_read_bytes: int = 0
    replay_h2d_transfer_count: int = 0
    replay_h2d_transfer_bytes: int = 0
    provider_reexecution_count: int = 0
    capture_reexecution_count: int = 0
    vjp_reexecution_count: int = 0
    source_native_device_materialization: bool = True
    host_custody_replay: bool = False
    same_capture_and_vjp: bool = True
    corrected_moments_recomputed: bool = False
    rank_stability_authority: bool = False
    d_decision_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_DEVICE_FINAL_RANK_SOURCE_AUTHORITY
    schema: str = SAMPLED_DEVICE_FINAL_RANK_SOURCE_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledDeviceFinalRankSourceRefusalCode.AUTHORITY_VIOLATION,
                "device-final-rank receipts require their exact issuer",
            )
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            if item.name.endswith("_hash") or item.name in {
                "device_identity",
                "authority_scope",
                "schema",
            }:
                _identity(getattr(self, item.name), name=item.name)
        for name in (
            "requested_physical_rank",
            "registered_look_count",
            "packet_count",
            "context_count",
            "original_capture_batch_count",
            "original_vjp_count",
            "route_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        for name in (
            "final_route_scale_d2h_bytes",
            "host_custody_packet_read_count",
            "host_custody_payload_read_bytes",
            "replay_h2d_transfer_count",
            "replay_h2d_transfer_bytes",
            "provider_reexecution_count",
            "capture_reexecution_count",
            "vjp_reexecution_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        flags = (
            self.source_native_device_materialization,
            self.host_custody_replay,
            self.same_capture_and_vjp,
            self.corrected_moments_recomputed,
            self.rank_stability_authority,
            self.d_decision_authority,
            self.c_k_o_authority,
            self.mask_commit_authority,
            self.terminal_or_pareto_authority,
        )
        if (
            any(type(flag) is not bool for flag in flags)
            or flags != (True, False, True, False, False, False, False, False, False)
            or any(
                getattr(self, name) != 0
                for name in (
                    "host_custody_packet_read_count",
                    "host_custody_payload_read_bytes",
                    "replay_h2d_transfer_count",
                    "replay_h2d_transfer_bytes",
                    "provider_reexecution_count",
                    "capture_reexecution_count",
                    "vjp_reexecution_count",
                )
            )
            or self.original_capture_batch_count != self.packet_count
            or self.original_vjp_count != self.packet_count
            or self.registered_look_count > self.packet_count
            or self.literal_j_stream_identity_hash
            == self.corrected_a_stream_identity_hash
            or self.authority_scope != SAMPLED_DEVICE_FINAL_RANK_SOURCE_AUTHORITY
            or self.schema != SAMPLED_DEVICE_FINAL_RANK_SOURCE_SCHEMA
        ):
            _refuse(
                V1SampledDeviceFinalRankSourceRefusalCode.AUTHORITY_VIOLATION,
                "device-final-rank receipt mixed replay provenance or authority",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_RECEIPT_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and expected_hash != observed
        ):
            _refuse(
                V1SampledDeviceFinalRankSourceRefusalCode.STALE_SOURCE,
                "device-final-rank receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class V1SampledFDGeometrySourceCoordinate:
    provenance: V1SampledFDGeometrySourceProvenance
    provenance_receipt_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    corrected_moments_state_hash: str
    requested_physical_rank: int
    registered_look_count: int
    packet_count: int
    context_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    packet_axis_hash: str
    source_measurement_axis_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    literal_j_stream_identity_hash: str
    corrected_a_stream_identity_hash: str
    registered_d_prefix_hash: str | None
    source_native_device_materialization: bool
    host_custody_replay: bool
    schema: str = SAMPLED_FD_GEOMETRY_SOURCE_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.provenance) is not V1SampledFDGeometrySourceProvenance:
            _refuse(
                V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
                "FD geometry provenance mode must be exact",
            )
        for item in fields(self):
            if item.name.endswith("_hash") and getattr(self, item.name) is not None:
                _identity(getattr(self, item.name), name=item.name)
        for name in (
            "requested_physical_rank",
            "registered_look_count",
            "packet_count",
            "context_count",
            "route_count",
        ):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        expected = (
            self.provenance
            is V1SampledFDGeometrySourceProvenance.DEVICE_RESIDENT_FINAL_RANK
        )
        if (
            self.source_native_device_materialization is not expected
            or self.host_custody_replay is expected
            or (
                expected
                and self.registered_d_prefix_hash is None
            )
            or self.literal_j_stream_identity_hash
            == self.corrected_a_stream_identity_hash
            or self.schema != SAMPLED_FD_GEOMETRY_SOURCE_COORDINATE_SCHEMA
        ):
            _refuse(
                V1SampledDeviceFinalRankSourceRefusalCode.MIXED_PROVENANCE,
                "FD geometry coordinate mixed host and device provenance",
            )

    @property
    def coordinate_hash(self) -> str:
        return content_hash(self, prefix=_COORDINATE_PREFIX)


def build_v1_sampled_device_registered_d_prefix_binding(
    *,
    registered_d_transaction: V1SampledCompositionRegisteredDTransaction,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    row_prefix_attachment: V1SampledDualChartRowLivePrefixAttachment,
    corrected_moments: V1StratifiedCorrectedAttributionMoments,
) -> V1SampledCompositionRegisteredDPrefixBinding:
    """Bind the registered request view to the realized same-capture axes."""

    if (
        type(registered_d_transaction)
        is not V1SampledCompositionRegisteredDTransaction
        or type(original_capture_binding)
        is not V1SampledLiteralJOriginalCaptureBinding
        or type(row_prefix_attachment)
        is not V1SampledDualChartRowLivePrefixAttachment
        or type(corrected_moments)
        is not V1StratifiedCorrectedAttributionMoments
    ):
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
            "registered-D prefix binding requires exact registered/realized owners",
        )
    original_capture_binding.assert_integrity()
    row_prefix_attachment.verify_live_witnesses()
    corrected_moments.assert_integrity()
    manifest = row_prefix_attachment.manifest
    flat_contexts = _flatten(original_capture_binding.packet_context_ids)
    flat_draws = _flatten(original_capture_binding.packet_draw_ids)
    flat_measurements = _flatten(
        original_capture_binding.packet_fd_source_measurement_hashes
    )
    if (
        manifest.manifest_hash
        != original_capture_binding.row_prefix_manifest_hash
        or len(flat_contexts) != len(flat_draws)
        or len(flat_contexts) != len(flat_measurements)
        or len(flat_contexts) != sum(corrected_moments.stratum_counts)
    ):
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.SOURCE_MISMATCH,
            "registered-D realized prefix axes do not close",
        )
    context_hash = content_hash(flat_contexts, prefix=_CONTEXT_AXIS_PREFIX)
    draw_hash = content_hash(flat_draws, prefix=_DRAW_AXIS_PREFIX)
    measurement_hash = content_hash(
        flat_measurements,
        prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
    )
    weight_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_device_D_unweighted_prefix_v1",
            "stratum_counts": tuple(corrected_moments.stratum_counts),
            "packet_stratum_labels": (
                original_capture_binding.packet_stratum_labels
            ),
            "unweighted_within_stratum": True,
        },
        prefix=_PREFIX + "unweighted-prefix-",
    )
    return build_v1_sampled_composition_registered_d_prefix_binding(
        rank_invariant_raw_prefix_coordinate=(
            registered_d_transaction.rank_invariant_raw_prefix_coordinate
        ),
        registered_look_count=registered_d_transaction.registered_look_count,
        literal_j_source_receipt_hash=(
            registered_d_transaction.literal_j_source_receipt_hash
        ),
        current_mask_hash=manifest.current_mask_hash,
        route_axis_hash=manifest.route_axis_hash,
        route_count=manifest.route_count,
        proposal_generation=manifest.proposal_generation,
        cumulative_stratum_counts=tuple(corrected_moments.stratum_counts),
        D_context_axis_hash=context_hash,
        D_draw_axis_hash=draw_hash,
        D_source_measurement_axis_hash=measurement_hash,
        D_weight_hash=weight_hash,
    )


def issue_v1_sampled_device_final_rank_source(
    *,
    materialization: V1SampledDeviceResidentDSessionMaterialization,
    materialization_rank_receipt: V1SampledDeviceResidentDMaterializationRankReceipt,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    row_prefix_attachment: V1SampledDualChartRowLivePrefixAttachment,
    fd_design: V1StratifiedFDDesign,
    registered_d_transaction: V1SampledCompositionRegisteredDTransaction,
    registered_d_prefix_binding: V1SampledCompositionRegisteredDPrefixBinding,
    device_identity: str,
    registered_d_prefix_v2: V1FQREndogenousRankDPrefixV2 | None = None,
) -> V1SampledDeviceFinalRankSourceReceipt:
    """Issue the device-native source from exact endpoint-owned objects.

    ``registered_d_prefix_v2``: when the row
    prefix has also been registered as a v2 endogenous-rank D prefix whose
    ``registered_d_prefix_identity_hash`` is exactly this endpoint's device
    prefix identity (``registered_d_prefix_binding.binding_hash``) and whose
    realized context/draw axes and observed stratum counts replay the same
    capture, the receipt's ``registered_d_prefix_hash`` is the v2
    ``prefix_hash`` so the device source and the v2 prefix name one identity.
    Without it the legacy row-manifest prefix hash is retained unchanged.
    """

    exact = (
        (materialization, V1SampledDeviceResidentDSessionMaterialization),
        (
            materialization_rank_receipt,
            V1SampledDeviceResidentDMaterializationRankReceipt,
        ),
        (original_capture_binding, V1SampledLiteralJOriginalCaptureBinding),
        (row_prefix_attachment, V1SampledDualChartRowLivePrefixAttachment),
        (fd_design, V1StratifiedFDDesign),
        (registered_d_transaction, V1SampledCompositionRegisteredDTransaction),
        (
            registered_d_prefix_binding,
            V1SampledCompositionRegisteredDPrefixBinding,
        ),
    )
    if any(type(value) is not expected for value, expected in exact):
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
            "device-final-rank issuance requires exact endpoint objects",
        )
    original_capture_binding.assert_integrity()
    row_prefix_attachment.verify_live_witnesses()
    manifest = row_prefix_attachment.manifest
    snapshots = tuple(materialization.fd_snapshots_by_rank)
    if len(snapshots) != 1:
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.SOURCE_MISMATCH,
            "device-final-rank source requires exactly one FD snapshot",
        )
    rank, snapshot = snapshots[0]
    snapshot.assert_integrity()
    moments = materialization.corrected_attribution_moments
    moments.assert_integrity()
    looks = tuple(manifest.looks)
    packet_count = len(looks)
    context_count = len(manifest.context_ids)
    raw_prefix = registered_d_transaction.rank_invariant_raw_prefix_coordinate
    closure = {
        "materialization_mode": materialization.materialization_mode
        == DEVICE_RESIDENT_D_REGISTERED_RANK_MATERIALIZATION_MODE,
        "materialization_rank": materialization.materialization_rank_receipt_hash
        == materialization_rank_receipt.receipt_hash,
        "session_design": materialization.session_design_hash
        == materialization_rank_receipt.session_design_hash,
        "hbm_forecast": materialization.hbm_forecast_hash
        == materialization_rank_receipt.hbm_forecast_hash,
        "selected_rank": materialization.selected_physical_fd_rank == rank,
        "registered_rank": (
            materialization_rank_receipt.registered_proposal_fd_rank == rank
        ),
        "transaction_rank": registered_d_transaction.physical_fd_rank == rank,
        "transaction_design": (
            registered_d_transaction.fd_design_hash == fd_design.design_hash
        ),
        "prefix_coordinate": registered_d_transaction.rank_invariant_raw_prefix_hash
        == registered_d_prefix_binding.rank_invariant_raw_prefix_hash,
        "prefix_source": registered_d_transaction.literal_j_source_receipt_hash
        == registered_d_prefix_binding.literal_j_source_receipt_hash,
        "prefix_binding_hash": registered_d_transaction.source_replay_axis_binding_hash
        == registered_d_prefix_binding.binding_hash,
        # The materialization counter is a registered statistical-look count;
        # ``manifest.looks`` is packet-granular because one look may be split
        # into several bounded train-provider calls.
        "materialization_looks": materialization.final_look_count
        == registered_d_transaction.registered_look_count,
        "moment_look": moments.look_index
        == registered_d_transaction.registered_look_count,
        "transaction_looks": (
            registered_d_transaction.registered_look_count
            == registered_d_prefix_binding.registered_look_count
            <= packet_count
        ),
        "snapshot_design": snapshot.design_hash == fd_design.design_hash,
        "snapshot_rank": snapshot.sketch_rank == rank,
        "snapshot_packets": snapshot.packet_boundary_index == packet_count,
        "materialization_route_axis": (
            materialization.route_axis_hash == fd_design.route_axis_hash
        ),
        "binding_manifest": manifest.manifest_hash
        == original_capture_binding.row_prefix_manifest_hash,
        "binding_route_ids": manifest.route_id_axis_hash
        == original_capture_binding.route_id_axis_hash,
        "binding_route_axis": manifest.route_axis_hash
        == original_capture_binding.route_axis_hash,
        "route_count": manifest.route_count == len(fd_design.route_ids),
        "capture_count": (
            original_capture_binding.original_capture_batch_count == packet_count
        ),
        "vjp_count": original_capture_binding.original_vjp_count == packet_count,
        "moment_context_count": context_count == sum(moments.stratum_counts),
        "snapshot_context_count": context_count
        == sum(child.context_count for _name, child in snapshot.stratum_snapshots),
        "corrected_stream": moments.raw_logit_stream_source_hash
        == manifest.corrected_a_stream_identity_hash,
        "literal_stream": fd_design.raw_logit_stream_source_hash
        == manifest.literal_j_stream_identity_hash,
        "prefix_mask": registered_d_prefix_binding.current_mask_hash
        == manifest.current_mask_hash,
        "prefix_generation": registered_d_prefix_binding.proposal_generation
        == manifest.proposal_generation,
        "prefix_route_axis": registered_d_prefix_binding.route_axis_hash
        == manifest.route_axis_hash,
        "prefix_route_count": registered_d_prefix_binding.route_count
        == manifest.route_count,
        "prefix_stratum_counts": tuple(
            registered_d_prefix_binding.cumulative_stratum_counts
        )
        == tuple(moments.stratum_counts),
        "raw_prefix_shape": len(raw_prefix) >= 13,
        "raw_prefix_mask": len(raw_prefix) >= 13
        and raw_prefix[5] == manifest.current_mask_hash,
        "raw_prefix_generation": len(raw_prefix) >= 13
        and raw_prefix[8] == manifest.proposal_generation,
        "raw_prefix_corrected_stream": len(raw_prefix) >= 13
        and raw_prefix[11] == manifest.corrected_a_stream_identity_hash,
        "raw_prefix_literal_stream": len(raw_prefix) >= 13
        and raw_prefix[12] == manifest.literal_j_stream_identity_hash,
    }
    if not all(closure.values()):
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.SOURCE_MISMATCH,
            "device-final-rank endpoint coordinates do not close; failed="
            f"{tuple(name for name, closed in closure.items() if not closed)}",
        )
    flat_contexts = _flatten(original_capture_binding.packet_context_ids)
    flat_draws = _flatten(original_capture_binding.packet_draw_ids)
    flat_strata = _flatten(original_capture_binding.packet_stratum_labels)
    flat_measurements = _flatten(
        original_capture_binding.packet_fd_source_measurement_hashes
    )
    context_axis_hash = content_hash(flat_contexts, prefix=_CONTEXT_AXIS_PREFIX)
    draw_axis_hash = content_hash(flat_draws, prefix=_DRAW_AXIS_PREFIX)
    source_measurement_axis_hash = content_hash(
        flat_measurements,
        prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
    )
    if (
        registered_d_prefix_binding.D_context_axis_hash != context_axis_hash
        or registered_d_prefix_binding.D_draw_axis_hash != draw_axis_hash
        or registered_d_prefix_binding.D_source_measurement_axis_hash
        != source_measurement_axis_hash
    ):
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.SOURCE_MISMATCH,
            "registered-D prefix binding differs from realized capture axes",
        )
    registered_d_prefix_hash = manifest.registered_d_prefix_hash
    if registered_d_prefix_v2 is not None:
        if type(registered_d_prefix_v2) is not V1FQREndogenousRankDPrefixV2:
            _refuse(
                V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
                "device-final-rank issuance requires the exact v2 D prefix",
            )
        registered_d_prefix_v2.assert_integrity()
        observed_by_stratum = dict(registered_d_prefix_v2.observed_stratum_counts)
        expected_by_stratum = {
            name: int(count)
            for name, count in zip(
                fd_design.stratum_names,
                moments.stratum_counts,
                strict=True,
            )
        }
        v2_closure = {
            "identity": registered_d_prefix_v2.registered_d_prefix_identity_hash
            == registered_d_prefix_binding.binding_hash,
            "mask": registered_d_prefix_v2.current_mask_hash
            == manifest.current_mask_hash,
            "route_axis": registered_d_prefix_v2.route_axis_hash
            == manifest.route_axis_hash,
            "generation": registered_d_prefix_v2.proposal_generation
            == manifest.proposal_generation,
            "context_axis": registered_d_prefix_v2.D_context_axis_hash
            == context_axis_hash,
            "draw_axis": registered_d_prefix_v2.D_draw_axis_hash == draw_axis_hash,
            "observed_counts": observed_by_stratum == expected_by_stratum,
        }
        if not all(v2_closure.values()):
            _refuse(
                V1SampledDeviceFinalRankSourceRefusalCode.SOURCE_MISMATCH,
                "v2 endogenous-rank D prefix does not name this device prefix "
                f"identity; failed={tuple(k for k, v in v2_closure.items() if not v)}",
            )
        registered_d_prefix_hash = registered_d_prefix_v2.prefix_hash
    packet_axis = tuple(
        (
            original_capture_binding.packet_plan_hashes[index],
            original_capture_binding.packet_source_capture_identity_hashes[index],
            original_capture_binding.packet_provider_proof_hashes[index],
            original_capture_binding.packet_raw_dual_proof_hashes[index],
            look.literal_j_shard.receipt_hash,
        )
        for index, look in enumerate(looks)
    )
    return V1SampledDeviceFinalRankSourceReceipt(
        original_capture_binding_hash=original_capture_binding.binding_hash,
        device_materialization_hash=materialization.materialization_hash,
        materialization_rank_receipt_hash=materialization_rank_receipt.receipt_hash,
        session_design_hash=materialization.session_design_hash,
        hbm_forecast_hash=materialization.hbm_forecast_hash,
        registered_d_transaction_hash=registered_d_transaction.transaction_hash,
        registered_d_prefix_binding_hash=registered_d_prefix_binding.binding_hash,
        rank_invariant_raw_prefix_hash=(
            registered_d_transaction.rank_invariant_raw_prefix_hash
        ),
        literal_j_source_receipt_hash=(
            registered_d_transaction.literal_j_source_receipt_hash
        ),
        row_prefix_manifest_hash=manifest.manifest_hash,
        fd_design_hash=fd_design.design_hash,
        fd_snapshot_hash=snapshot.snapshot_hash,
        corrected_moments_state_hash=moments.state_hash,
        requested_physical_rank=rank,
        registered_look_count=registered_d_transaction.registered_look_count,
        packet_count=packet_count,
        context_count=context_count,
        original_capture_batch_count=(
            original_capture_binding.original_capture_batch_count
        ),
        original_vjp_count=original_capture_binding.original_vjp_count,
        context_axis_hash=context_axis_hash,
        draw_axis_hash=draw_axis_hash,
        stratum_axis_hash=content_hash(flat_strata, prefix=_STRATUM_AXIS_PREFIX),
        packet_axis_hash=content_hash(packet_axis, prefix=_PACKET_AXIS_PREFIX),
        source_measurement_axis_hash=source_measurement_axis_hash,
        registered_d_prefix_hash=registered_d_prefix_hash,
        chart_identity_hash=manifest.chart_identity_hash,
        current_mask_hash=manifest.current_mask_hash,
        proposal_generation=manifest.proposal_generation,
        route_axis_hash=manifest.route_axis_hash,
        route_id_axis_hash=manifest.route_id_axis_hash,
        route_count=manifest.route_count,
        literal_j_stream_identity_hash=manifest.literal_j_stream_identity_hash,
        corrected_a_stream_identity_hash=manifest.corrected_a_stream_identity_hash,
        device_identity=_identity(device_identity, name="device_identity"),
        final_route_scale_d2h_bytes=materialization.final_route_scale_d2h_bytes,
        _issuance_token=_ISSUER_TOKEN,
    )


def load_v1_sampled_device_final_rank_source(
    payload: Mapping[str, object],
    *,
    expected_receipt_hash: str | None = None,
) -> V1SampledDeviceFinalRankSourceReceipt:
    if not isinstance(payload, Mapping):
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
            "device-final-rank payload must be one mapping",
        )
    values = dict(payload)
    declared = values.pop("receipt_hash", None)
    allowed = {
        item.name
        for item in fields(V1SampledDeviceFinalRankSourceReceipt)
        if not item.name.startswith("_")
    }
    if set(values) != allowed:
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.INVALID_TYPE,
            "device-final-rank payload fields do not match its schema",
        )
    receipt = V1SampledDeviceFinalRankSourceReceipt(
        **values,
        _issuance_token=_ISSUER_TOKEN,
    )
    expected = expected_receipt_hash
    if expected is None and declared is not None:
        expected = _identity(declared, name="declared receipt hash")
    receipt.assert_integrity(expected)
    return receipt


def normalize_v1_sampled_fd_geometry_source(
    source: V1SampledLiteralJFDRankReplayReceipt
    | V1SampledDeviceFinalRankSourceReceipt,
) -> V1SampledFDGeometrySourceCoordinate:
    """Project an exact disjoint provenance source onto scientific fields."""

    if type(source) is V1SampledLiteralJFDRankReplayReceipt:
        source.assert_integrity()
        provenance = V1SampledFDGeometrySourceProvenance.HOST_CUSTODY_RANK_REPLAY
        registered_prefix = None
        registered_look_count = source.packet_count
        device = False
        host = True
    elif type(source) is V1SampledDeviceFinalRankSourceReceipt:
        source.assert_integrity()
        provenance = V1SampledFDGeometrySourceProvenance.DEVICE_RESIDENT_FINAL_RANK
        registered_prefix = source.registered_d_prefix_hash
        registered_look_count = source.registered_look_count
        device = True
        host = False
    else:
        _refuse(
            V1SampledDeviceFinalRankSourceRefusalCode.MIXED_PROVENANCE,
            "FD geometry source must be one exact host or device receipt",
        )
    return V1SampledFDGeometrySourceCoordinate(
        provenance=provenance,
        provenance_receipt_hash=source.receipt_hash,
        fd_design_hash=source.fd_design_hash,
        fd_snapshot_hash=source.fd_snapshot_hash,
        corrected_moments_state_hash=source.corrected_moments_state_hash,
        requested_physical_rank=source.requested_physical_rank,
        registered_look_count=registered_look_count,
        packet_count=source.packet_count,
        context_count=source.context_count,
        context_axis_hash=source.context_axis_hash,
        draw_axis_hash=source.draw_axis_hash,
        stratum_axis_hash=source.stratum_axis_hash,
        packet_axis_hash=source.packet_axis_hash,
        source_measurement_axis_hash=source.source_measurement_axis_hash,
        chart_identity_hash=source.chart_identity_hash,
        current_mask_hash=source.current_mask_hash,
        proposal_generation=source.proposal_generation,
        route_axis_hash=source.route_axis_hash,
        route_id_axis_hash=source.route_id_axis_hash,
        route_count=source.route_count,
        literal_j_stream_identity_hash=source.literal_j_stream_identity_hash,
        corrected_a_stream_identity_hash=source.corrected_a_stream_identity_hash,
        registered_d_prefix_hash=registered_prefix,
        source_native_device_materialization=device,
        host_custody_replay=host,
    )


__all__ = [
    "SAMPLED_DEVICE_FINAL_RANK_SOURCE_AUTHORITY",
    "SAMPLED_DEVICE_FINAL_RANK_SOURCE_SCHEMA",
    "SAMPLED_FD_GEOMETRY_SOURCE_COORDINATE_SCHEMA",
    "V1SampledDeviceFinalRankSourceReceipt",
    "V1SampledDeviceFinalRankSourceRefusal",
    "V1SampledDeviceFinalRankSourceRefusalCode",
    "V1SampledFDGeometrySourceCoordinate",
    "V1SampledFDGeometrySourceProvenance",
    "build_v1_sampled_device_registered_d_prefix_binding",
    "issue_v1_sampled_device_final_rank_source",
    "load_v1_sampled_device_final_rank_source",
    "normalize_v1_sampled_fd_geometry_source",
]
