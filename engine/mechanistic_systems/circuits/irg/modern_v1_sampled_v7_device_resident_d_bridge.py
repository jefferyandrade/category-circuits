"""Atomic sampled-V7 bridge from one D look to live state plus exact rows.

One provider capture/VJP is fanned out to two noninterchangeable owners:

* the device-resident corrected-A Welford and literal-J FD session; and
* the existing immutable corrected-A/literal-J row-custody transaction.

The row HEAD is published only after every packet, provider proof, resident
update, and custody proof closes.  Any earlier failure aborts the row group and
evicts the non-invertible resident session, forcing restart from the sealed D
prefix.  The path-free boundary contains no route-scale arrays and grants no
C/K/O, mask-commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.optimized_irg_providers import (
    build_route_chunked_dual_chart_fanout,
)

from .modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDLookReceipt,
    V1SampledDeviceResidentDSession,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowGroupPlan,
    V1SampledDualChartRowLivePrefixAttachment,
    begin_v1_sampled_dual_chart_row_group_transaction,
)
from .modern_v1_sampled_look_transaction import (
    SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6,
    V1SampledDeviceResidentDTransactionResult,
    V1SampledDualChartChunkObjectSeparationWitness,
    V1SampledDualChartRegisteredLookReceipt,
    V1SampledDualChartRowCustodyRegisteredLookReceipt,
    V1SampledDualRowGroupPacketBridge,
    V1SampledRegisteredLookTransactionRequest,
    _DeviceResidentDSessionPacketBridge,
    _provider_plan_identity_hash,
    _source_capture_identity_hash,
    _verify_device_resident_d_provider_receipt,
    verify_v1_sampled_dual_chart_custody_only_provider_receipt,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJOriginalCaptureBinding,
)


SAMPLED_V7_DEVICE_RESIDENT_D_ROW_BOUNDARY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_device_resident_D_row_boundary_v1"
)
SAMPLED_V7_DEVICE_RESIDENT_D_ROW_AUTHORITY: Final = (
    "D_sufficient_statistics_and_exact_dual_row_custody_only_no_C_K_O_"
    "commit_terminal_or_pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-sampled-v7-device-resident-D-row-"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _exact_branch_children(
    observed: object,
    *,
    resident: Mapping[str, object],
    custody: Mapping[str, object],
    role: str,
) -> None:
    if not isinstance(observed, Mapping):
        raise TypeError(f"{role} provider fanout receipt is not a mapping")
    children = tuple(observed.get("loss_only_stream_tee_child_receipts", ()))
    if (
        observed.get("loss_only_stream_tee") is not True
        or observed.get("loss_only_stream_tee_child_count") != 2
        or observed.get("shared_positive_removal_rows") is not True
        or len(children) != 2
        or not all(isinstance(value, Mapping) for value in children)
        or tuple(dict(value) for value in children)
        != (dict(resident), dict(custody))
    ):
        raise ValueError(
            f"{role} provider branch did not tee one exact row object into "
            "resident and custody owners"
        )


def _project_direct_receipt(
    receipt: Mapping[str, object],
    *,
    evidence: Mapping[str, object],
    jacobian: Mapping[str, object],
    custody: bool,
) -> Mapping[str, object]:
    projected = dict(receipt)
    projected["dual_chart_evidence_receipt"] = dict(evidence)
    projected["dual_chart_jacobian_receipt"] = dict(jacobian)
    projected["loss_only_stream_tee"] = False
    projected["loss_only_stream_tee_child_count"] = 0
    projected["loss_only_stream_tee_child_receipts"] = ()
    projected["shared_positive_removal_rows"] = False
    if not custody:
        projected["host_context_route_rows_transferred"] = False
        projected["host_context_route_row_transfer_count"] = 0
        projected["host_context_route_row_transfer_bytes"] = 0
        projected["host_context_route_transfer_role"] = "none"
    return MappingProxyType(projected)


@dataclass(frozen=True, slots=True)
class V1SampledV7DeviceResidentDRowLookBoundary:
    """Compact, restart-comparable result of one atomically published D look."""

    resident_result: V1SampledDeviceResidentDTransactionResult
    dual_chart_receipt: V1SampledDualChartRegisteredLookReceipt
    row_prefix_manifest_hash: str
    row_packet_look_hashes: tuple[str, ...]
    row_role_payload_sha256_pairs: tuple[tuple[str, str], ...]
    source_capture_identity_hashes: tuple[str, ...]
    prior_boundary_hash: str | None = None
    intermediate_moment_canonicalization_count: int = 0
    intermediate_fd_materialization_count: int = 0
    route_scale_moment_d2h_bytes: int = 0
    provider_vjp_count: int = 0
    authority_scope: str = SAMPLED_V7_DEVICE_RESIDENT_D_ROW_AUTHORITY
    C_authority: bool = False
    K_authority: bool = False
    O_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_V7_DEVICE_RESIDENT_D_ROW_BOUNDARY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.resident_result) is not V1SampledDeviceResidentDTransactionResult:
            raise TypeError("V7 D-row boundary requires its exact resident result")
        if type(self.dual_chart_receipt) is not V1SampledDualChartRegisteredLookReceipt:
            raise TypeError("V7 D-row boundary requires its exact dual receipt")
        manifest = _identity(
            self.row_prefix_manifest_hash,
            name="V7 D-row prefix manifest hash",
        )
        row_looks = tuple(
            _identity(value, name="V7 D-row packet look hash")
            for value in self.row_packet_look_hashes
        )
        captures = tuple(
            _identity(value, name="V7 D-row source-capture identity")
            for value in self.source_capture_identity_hashes
        )
        payload_pairs = tuple(
            (
                _identity(pair[0], name="corrected-A payload sha256"),
                _identity(pair[1], name="literal-J payload sha256"),
            )
            for pair in self.row_role_payload_sha256_pairs
        )
        packet_count = len(self.resident_result.look_receipts)
        if (
            packet_count <= 0
            or len(row_looks) != packet_count
            or len(captures) != packet_count
            or len(payload_pairs) != packet_count
            or self.dual_chart_receipt.packet_count != packet_count
            or self.dual_chart_receipt.transaction_request_hash
            != self.resident_result.transaction_request_hash
            or self.provider_vjp_count != packet_count
            or len(set(row_looks)) != len(row_looks)
            or len(set(captures)) != len(captures)
        ):
            raise ValueError("V7 D-row boundary has a partial packet/VJP ledger")
        if self.prior_boundary_hash is not None:
            _identity(self.prior_boundary_hash, name="prior V7 D-row boundary hash")
        flags = (
            self.C_authority,
            self.K_authority,
            self.O_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            self.intermediate_moment_canonicalization_count != 0
            or self.intermediate_fd_materialization_count != 0
            or self.route_scale_moment_d2h_bytes != 0
            or self.authority_scope != SAMPLED_V7_DEVICE_RESIDENT_D_ROW_AUTHORITY
            or any(type(value) is not bool for value in flags)
            or any(flags)
            or self.schema != SAMPLED_V7_DEVICE_RESIDENT_D_ROW_BOUNDARY_SCHEMA
        ):
            raise PermissionError("V7 D-row boundary crossed its authority boundary")
        object.__setattr__(self, "row_prefix_manifest_hash", manifest)
        object.__setattr__(self, "row_packet_look_hashes", row_looks)
        object.__setattr__(self, "source_capture_identity_hashes", captures)
        object.__setattr__(self, "row_role_payload_sha256_pairs", payload_pairs)

    @property
    def boundary_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "resident_result_hash": self.resident_result.result_hash,
                "dual_chart_receipt_hash": self.dual_chart_receipt.receipt_hash,
                # Manifest/look hashes also bind pre-outcome scratch forecasts.
                # Restart identity instead uses the exact role payload digests,
                # so operational free-space telemetry cannot change science.
                "row_role_payload_sha256_pairs": (
                    self.row_role_payload_sha256_pairs
                ),
                "source_capture_identity_hashes": (
                    self.source_capture_identity_hashes
                ),
                "prior_boundary_hash": self.prior_boundary_hash,
                "intermediate_moment_canonicalization_count": 0,
                "intermediate_fd_materialization_count": 0,
                "route_scale_moment_d2h_bytes": 0,
                "provider_vjp_count": self.provider_vjp_count,
                "authority_scope": self.authority_scope,
                "all_downstream_authority": False,
            },
            prefix=_PREFIX + "boundary-",
        )

    def to_json_payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "boundary_hash": self.boundary_hash,
                "resident_result_hash": self.resident_result.result_hash,
                "dual_chart_receipt_hash": self.dual_chart_receipt.receipt_hash,
                "row_prefix_manifest_hash": self.row_prefix_manifest_hash,
                "row_packet_look_hashes": self.row_packet_look_hashes,
                "row_role_payload_sha256_pairs": (
                    self.row_role_payload_sha256_pairs
                ),
                "source_capture_identity_hashes": (
                    self.source_capture_identity_hashes
                ),
                "prior_boundary_hash": self.prior_boundary_hash,
                "intermediate_moment_canonicalization_count": 0,
                "intermediate_fd_materialization_count": 0,
                "route_scale_moment_d2h_bytes": 0,
                "provider_vjp_count": self.provider_vjp_count,
                "authority_scope": self.authority_scope,
                "all_downstream_authority": False,
            }
        )


@dataclass(slots=True)
class V1SampledV7DeviceResidentDRowLookCommit:
    """Live published row attachment paired with one compact D boundary."""

    boundary: V1SampledV7DeviceResidentDRowLookBoundary
    row_custody_receipt: V1SampledDualChartRowCustodyRegisteredLookReceipt
    row_prefix_attachment: V1SampledDualChartRowLivePrefixAttachment = field(
        repr=False,
    )
    runtime_diagnostics: Mapping[str, object]
    head_published: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.boundary) is not V1SampledV7DeviceResidentDRowLookBoundary
            or type(self.row_custody_receipt)
            is not V1SampledDualChartRowCustodyRegisteredLookReceipt
            or type(self.row_prefix_attachment)
            is not V1SampledDualChartRowLivePrefixAttachment
        ):
            raise TypeError("V7 D-row commit types are not exact")
        self.row_prefix_attachment.verify_live_witnesses()
        if (
            self.row_prefix_attachment.manifest.manifest_hash
            != self.boundary.row_prefix_manifest_hash
            or self.row_custody_receipt.group_commit_receipt.manifest_hash
            != self.boundary.row_prefix_manifest_hash
            or type(self.head_published) is not bool
            or self.head_published is not False
        ):
            raise ValueError("V7 D-row prepared commit changed its row manifest")
        diagnostics = dict(self.runtime_diagnostics)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in diagnostics.values()
        ):
            raise ValueError("V7 D-row diagnostics must be finite nonnegative scalars")
        self.runtime_diagnostics = MappingProxyType(diagnostics)

    def acknowledge_head_publication(
        self,
        attachment: V1SampledDualChartRowLivePrefixAttachment,
    ) -> None:
        if self.head_published:
            raise RuntimeError("V7 D-row HEAD publication was already acknowledged")
        if type(attachment) is not V1SampledDualChartRowLivePrefixAttachment:
            raise TypeError("V7 D-row publication requires exact live attachment")
        attachment.verify_live_witnesses()
        if attachment.manifest.manifest_hash != self.boundary.row_prefix_manifest_hash:
            raise ValueError("published V7 D-row attachment changed its manifest")
        self.row_prefix_attachment = attachment
        self.head_published = True

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("V7 D-row live commits are nonserializable")


def derive_v1_sampled_v7_original_capture_binding(
    *,
    request: V1SampledRegisteredLookTransactionRequest,
    commit: V1SampledV7DeviceResidentDRowLookCommit,
    prior_original_capture_binding: (
        V1SampledLiteralJOriginalCaptureBinding | None
    ) = None,
) -> V1SampledLiteralJOriginalCaptureBinding:
    """Derive cumulative same-capture custody directly from one live V7 look.

    This is not a V5 proof adapter.  It reads the V6 transaction plan, the
    atomically closed resident/custody receipt pair, and the immutable row
    prefix produced by that same call.  No provider, capture, VJP, row replay,
    or route-scale materialization occurs here.
    """

    if type(request) is not V1SampledRegisteredLookTransactionRequest:
        raise TypeError("V7 original-capture derivation requires its exact request")
    if request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6:
        raise ValueError("V7 original-capture derivation requires V6 lineage")
    if type(commit) is not V1SampledV7DeviceResidentDRowLookCommit:
        raise TypeError("V7 original-capture derivation requires its exact commit")
    if (
        prior_original_capture_binding is not None
        and type(prior_original_capture_binding)
        is not V1SampledLiteralJOriginalCaptureBinding
    ):
        raise TypeError("V7 prior original-capture binding is not exact")
    group_plan = request.dual_chart_row_group_plan
    if type(group_plan) is not V1SampledDualChartRowGroupPlan:
        raise TypeError("V7 original-capture derivation lacks its row-group plan")
    group_plan.assert_integrity()
    commit.row_prefix_attachment.verify_live_witnesses()
    manifest = commit.row_prefix_attachment.manifest
    custody = commit.row_custody_receipt
    dual = commit.boundary.dual_chart_receipt
    packet_plans = tuple(plan.plan_hash for plan in request.packet_plans)
    group_packets = tuple(group_plan.packet_plans)
    suffix_start = group_plan.first_prefix_look_ordinal
    suffix_stop = suffix_start + len(group_packets)
    suffix_looks = tuple(manifest.looks[suffix_start:suffix_stop])
    if (
        len(manifest.looks) != suffix_stop
        or manifest.manifest_hash != custody.group_commit_receipt.manifest_hash
        or manifest.manifest_hash != commit.boundary.row_prefix_manifest_hash
        or packet_plans != tuple(custody.registered_packet_plan_hashes)
        or packet_plans != tuple(
            packet.registered_packet_plan_hash for packet in group_packets
        )
        or packet_plans != tuple(dual.packet_plan_hashes)
        or len(suffix_looks) != len(group_packets)
        or tuple(
            packet.source_capture_identity_hash for packet in group_packets
        )
        != tuple(
            look.literal_j_shard.source_capture_identity_hash
            for look in suffix_looks
        )
        or tuple(packet.context_ids for packet in group_packets)
        != tuple(look.context_ids for look in suffix_looks)
        or tuple(packet.draw_ids for packet in group_packets)
        != tuple(look.draw_ids for look in suffix_looks)
        or tuple(packet.stratum_labels for packet in group_packets)
        != tuple(look.stratum_labels for look in suffix_looks)
    ):
        raise ValueError("V7 original-capture suffix proof does not close")
    shared = (
        (manifest.registered_d_prefix_hash, group_plan.registered_d_prefix_hash),
        (manifest.chart_identity_hash, group_plan.chart_identity_hash),
        (manifest.current_mask_hash, group_plan.current_mask_hash),
        (manifest.proposal_generation, group_plan.proposal_generation),
        (manifest.route_axis_hash, group_plan.route_axis_hash),
        (manifest.route_id_axis_hash, group_plan.route_id_axis_hash),
        (manifest.route_count, group_plan.route_count),
        (
            manifest.corrected_a_stream_identity_hash,
            group_plan.corrected_a_stream_identity_hash,
        ),
        (
            manifest.literal_j_stream_identity_hash,
            group_plan.literal_j_stream_identity_hash,
        ),
    )
    if any(left != right for left, right in shared):
        raise ValueError("V7 original-capture chart coordinates changed")

    prior = prior_original_capture_binding
    if prior is None:
        if suffix_start != 0 or group_plan.expected_prior_manifest_hash is not None:
            raise ValueError("noninitial V7 custody requires its exact prior binding")
        prior_packet_plans: tuple[str, ...] = ()
        prior_captures: tuple[str, ...] = ()
        prior_providers: tuple[str, ...] = ()
        prior_raw_dual: tuple[str, ...] = ()
        prior_measurements: tuple[tuple[str, ...], ...] = ()
        prior_contexts: tuple[tuple[str, ...], ...] = ()
        prior_draws: tuple[tuple[str, ...], ...] = ()
        prior_strata: tuple[tuple[str, ...], ...] = ()
        parent_hash = None
    else:
        prior_hash = prior.assert_integrity()
        prior_looks = tuple(manifest.looks[:suffix_start])
        if (
            suffix_start != len(prior.packet_plan_hashes)
            or group_plan.expected_prior_manifest_hash
            != prior.row_prefix_manifest_hash
            or custody.group_commit_receipt.prior_manifest_hash
            != prior.row_prefix_manifest_hash
            or tuple(look.context_ids for look in prior_looks)
            != prior.packet_context_ids
            or tuple(look.draw_ids for look in prior_looks)
            != prior.packet_draw_ids
            or tuple(look.stratum_labels for look in prior_looks)
            != prior.packet_stratum_labels
            or tuple(
                look.literal_j_shard.source_capture_identity_hash
                for look in prior_looks
            )
            != prior.packet_source_capture_identity_hashes
        ):
            raise ValueError("V7 original-capture predecessor binding changed")
        prior_shared = (
            (prior.registered_d_prefix_hash, manifest.registered_d_prefix_hash),
            (prior.chart_identity_hash, manifest.chart_identity_hash),
            (prior.current_mask_hash, manifest.current_mask_hash),
            (prior.proposal_generation, manifest.proposal_generation),
            (prior.route_axis_hash, manifest.route_axis_hash),
            (prior.route_id_axis_hash, manifest.route_id_axis_hash),
            (prior.route_count, manifest.route_count),
            (
                prior.corrected_a_stream_identity_hash,
                manifest.corrected_a_stream_identity_hash,
            ),
            (
                prior.literal_j_stream_identity_hash,
                manifest.literal_j_stream_identity_hash,
            ),
        )
        if any(left != right for left, right in prior_shared):
            raise ValueError("V7 original-capture predecessor crossed charts")
        prior_packet_plans = prior.packet_plan_hashes
        prior_captures = prior.packet_source_capture_identity_hashes
        prior_providers = prior.packet_provider_proof_hashes
        prior_raw_dual = prior.packet_raw_dual_proof_hashes
        prior_measurements = prior.packet_fd_source_measurement_hashes
        prior_contexts = prior.packet_context_ids
        prior_draws = prior.packet_draw_ids
        prior_strata = prior.packet_stratum_labels
        parent_hash = prior_hash

    binding = V1SampledLiteralJOriginalCaptureBinding(
        original_outer_receipt_hash=commit.boundary.boundary_hash,
        row_group_plan_hash=group_plan.group_plan_hash,
        row_prefix_manifest_hash=manifest.manifest_hash,
        parent_binding_hash=parent_hash,
        packet_plan_hashes=prior_packet_plans + packet_plans,
        packet_source_capture_identity_hashes=prior_captures
        + tuple(packet.source_capture_identity_hash for packet in group_packets),
        packet_provider_proof_hashes=prior_providers
        + tuple(custody.provider_receipt_hashes),
        packet_raw_dual_proof_hashes=prior_raw_dual
        + tuple(dual.raw_dual_stream_receipt_hashes),
        packet_fd_source_measurement_hashes=prior_measurements
        + tuple(
            tuple(plan.fd_request.source_measurement_hashes)
            for plan in request.packet_plans
        ),
        packet_context_ids=prior_contexts
        + tuple(look.context_ids for look in suffix_looks),
        packet_draw_ids=prior_draws
        + tuple(look.draw_ids for look in suffix_looks),
        packet_stratum_labels=prior_strata
        + tuple(look.stratum_labels for look in suffix_looks),
        corrected_a_stream_identity_hash=(
            manifest.corrected_a_stream_identity_hash
        ),
        literal_j_stream_identity_hash=manifest.literal_j_stream_identity_hash,
        original_capture_batch_count=len(prior_packet_plans) + len(packet_plans),
        original_vjp_count=len(prior_packet_plans) + len(packet_plans),
        registered_d_prefix_hash=manifest.registered_d_prefix_hash,
        chart_identity_hash=manifest.chart_identity_hash,
        current_mask_hash=manifest.current_mask_hash,
        proposal_generation=manifest.proposal_generation,
        route_axis_hash=manifest.route_axis_hash,
        route_id_axis_hash=manifest.route_id_axis_hash,
        route_count=manifest.route_count,
    )
    binding.assert_integrity()
    return binding


def execute_v1_sampled_v7_device_resident_d_row_look(
    *,
    request: V1SampledRegisteredLookTransactionRequest,
    provider: Any,
    measurement_requests: Sequence[Any],
    session: V1SampledDeviceResidentDSession,
    row_scratch_directory: Any,
    prior_boundary: V1SampledV7DeviceResidentDRowLookBoundary | None = None,
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None = None,
) -> V1SampledV7DeviceResidentDRowLookCommit:
    """Execute one registered look with one provider call per physical packet."""

    if type(request) is not V1SampledRegisteredLookTransactionRequest:
        raise TypeError("V7 D-row execution requires its exact request")
    if request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6:
        raise ValueError("V7 D-row execution requires the V6 request schema")
    if type(session) is not V1SampledDeviceResidentDSession:
        raise TypeError("V7 D-row execution requires its exact live session")
    group_plan = request.dual_chart_row_group_plan
    if type(group_plan) is not V1SampledDualChartRowGroupPlan:
        raise TypeError("V7 D-row request lacks its exact row group plan")
    requests = tuple(measurement_requests)
    if len(requests) != len(request.packet_plans):
        raise ValueError("V7 D-row measurement-request ledger is partial")
    expected_provider_plans = tuple(
        packet.provider_plan_identity_hash for packet in group_plan.packet_plans
    )
    expected_capture_plans = tuple(
        packet.source_capture_identity_hash for packet in group_plan.packet_plans
    )
    if tuple(_provider_plan_identity_hash(value) for value in requests) != (
        expected_provider_plans
    ):
        raise ValueError(
            "V7 D-row measurement requests differ from preregistered provider plans"
        )
    if tuple(_source_capture_identity_hash(value) for value in requests) != (
        expected_capture_plans
    ):
        raise ValueError(
            "V7 D-row measurement requests differ from preregistered capture plans"
        )
    execute = getattr(provider, "accumulate_device_tripartite", None)
    if not callable(execute):
        raise TypeError("V7 D-row provider lacks device tripartite execution")

    prior_result = request.prior_device_resident_result
    if prior_result is None:
        if (
            request.registered_look_index != 1
            or prior_boundary is not None
            or prior_row_prefix is not None
            or session.look_receipts
            or any(session.stratum_counts)
            or group_plan.expected_prior_manifest_hash is not None
        ):
            raise ValueError("first V7 D-row look requires empty resident and row custody")
    else:
        if (
            type(prior_boundary) is not V1SampledV7DeviceResidentDRowLookBoundary
            or type(prior_row_prefix) is not V1SampledDualChartRowLivePrefixAttachment
        ):
            raise TypeError("continued V7 D-row look requires both compact priors")
        prior_row_prefix.verify_live_witnesses()
        if (
            prior_boundary.resident_result.result_hash != prior_result.result_hash
            or prior_boundary.row_prefix_manifest_hash
            != prior_row_prefix.manifest.manifest_hash
            or group_plan.expected_prior_manifest_hash
            != prior_boundary.row_prefix_manifest_hash
            or tuple(receipt.receipt_hash for receipt in session.look_receipts)
            != prior_result.cumulative_look_receipt_hashes
            or session.stratum_counts != prior_result.cumulative_stratum_counts_after
        ):
            raise ValueError("continued V7 D-row look changed resident/row lineage")

    first_plan = request.packet_plans[0]
    packet = first_plan.reduced_request.packet_request
    coordinates = {
        "FD design": (session.design.fd_design.design_hash, request.design.design_hash),
        "route axis": (session.design.route_axis_hash, packet.route_axis_hash),
        "route count": (session.design.route_count, packet.route_count),
        "corrected-A stream": (
            session.design.corrected_a_stream_identity_hash,
            first_plan.corrected_evidence_stream_identity_hash,
        ),
        "literal-J stream": (
            session.design.literal_j_stream_identity_hash,
            first_plan.literal_jacobian_stream_identity_hash,
        ),
        "population counts": (
            session.design.population_stratum_counts,
            packet.population_stratum_counts,
        ),
        "population orders": (
            session.design.sealed_population_order_hashes,
            packet.sealed_population_order_hashes,
        ),
        "route chunk": (
            session.design.registered_route_chunk_size,
            packet.registered_route_chunk_size,
        ),
    }
    for name, (left, right) in coordinates.items():
        if left != right:
            raise ValueError(f"V7 D-row session differs from request on {name}")

    transaction = begin_v1_sampled_dual_chart_row_group_transaction(
        plan=group_plan,
        scratch_directory=row_scratch_directory,
        prior=prior_row_prefix,
    )
    resident_look_receipts: list[V1SampledDeviceResidentDLookReceipt] = []
    resident_provider_proofs: list[str] = []
    custody_provider_proofs: list[str] = []
    corrected_resident_hashes: list[str] = []
    literal_resident_hashes: list[str] = []
    corrected_custody_hashes: list[str] = []
    literal_custody_hashes: list[str] = []
    raw_dual_hashes: list[str] = []
    separation_hashes: list[str] = []
    provider_seconds = 0.0
    started = time.perf_counter()
    try:
        for ordinal, (plan, measurement) in enumerate(
            zip(request.packet_plans, requests, strict=True)
        ):
            resident_bridge = _DeviceResidentDSessionPacketBridge(
                transaction_request_hash=request.request_hash,
                registered_look_index=request.registered_look_index,
                packet_ordinal=ordinal,
                plan=plan,
                session=session,
            )
            witness = V1SampledDualChartChunkObjectSeparationWitness()
            group_packet = group_plan.packet_plans[ordinal]
            custody_bridge = V1SampledDualRowGroupPacketBridge(
                group_plan_hash=group_plan.group_plan_hash,
                group_packet_plan=group_packet,
                staging_consumer=transaction.open_packet_staging_consumer(
                    packet_ordinal=ordinal
                ),
                route_ids=plan.reduced_request.route_ids,
                device=plan.reduced_request.declared_device,
                destination_transport_hash=(
                    plan.reduced_request.destination_transport_hash
                ),
                registered_route_chunk_size=(
                    plan.reduced_request.packet_request.registered_route_chunk_size
                ),
                corrected_a_stream_identity_hash=(
                    plan.corrected_evidence_stream_identity_hash
                ),
                literal_j_stream_identity_hash=(
                    plan.literal_jacobian_stream_identity_hash
                ),
                dual_chunk_observer=witness.observe,
            )
            fanout = build_route_chunked_dual_chart_fanout(
                evidence_consumers=(
                    resident_bridge.evidence_consumer,
                    custody_bridge.evidence_consumer,
                ),
                jacobian_consumers=(
                    resident_bridge.jacobian_consumer,
                    custody_bridge.jacobian_consumer,
                ),
            )
            provider_started = time.perf_counter()
            provider_receipt = execute(measurement, accumulator=fanout)
            provider_seconds += time.perf_counter() - provider_started
            if not isinstance(provider_receipt, Mapping):
                raise TypeError("V7 D-row provider returned no typed receipt")

            resident_a = dict(resident_bridge.branch_receipt("corrected_a"))
            resident_j = dict(resident_bridge.branch_receipt("literal_j"))
            custody_a = dict(custody_bridge.branch_receipt("corrected_a"))
            custody_j = dict(custody_bridge.branch_receipt("literal_j"))
            _exact_branch_children(
                provider_receipt.get("dual_chart_evidence_receipt"),
                resident=resident_a,
                custody=custody_a,
                role="corrected-A",
            )
            _exact_branch_children(
                provider_receipt.get("dual_chart_jacobian_receipt"),
                resident=resident_j,
                custody=custody_j,
                role="literal-J",
            )
            separation = witness.close(
                expected_pair_count=len(tuple(custody_a["route_windows"]))
            )
            resident_proof = _verify_device_resident_d_provider_receipt(
                _project_direct_receipt(
                    provider_receipt,
                    evidence=resident_a,
                    jacobian=resident_j,
                    custody=False,
                ),
                plan=plan,
                bridge=resident_bridge,
            )
            custody_proof = (
                verify_v1_sampled_dual_chart_custody_only_provider_receipt(
                    _project_direct_receipt(
                        provider_receipt,
                        evidence=custody_a,
                        jacobian=custody_j,
                        custody=True,
                    ),
                    group_plan=group_plan,
                    packet_ordinal=ordinal,
                    row_custody_bridge=custody_bridge,
                    row_object_separation_receipt_hash=separation,
                )
            )
            custody_bridge.seal()
            transaction.mark_packet_proofs_closed(
                packet_ordinal=ordinal,
                provider_proof_hash=custody_proof.provider_receipt_hash,
                reducer_proof_hashes=(
                    str(resident_a["typed_receipt_hash"]),
                    str(resident_j["typed_receipt_hash"]),
                ),
            )
            resident_look_receipts.append(resident_bridge.look_receipt)
            resident_provider_proofs.append(resident_proof)
            custody_provider_proofs.append(custody_proof.provider_receipt_hash)
            corrected_resident_hashes.append(str(resident_a["typed_receipt_hash"]))
            literal_resident_hashes.append(str(resident_j["typed_receipt_hash"]))
            corrected_custody_hashes.append(
                custody_proof.corrected_custody_branch_receipt_hash
            )
            literal_custody_hashes.append(
                custody_proof.literal_j_custody_branch_receipt_hash
            )
            raw_dual_hashes.append(custody_proof.raw_dual_stream_receipt_hash)
            separation_hashes.append(separation)

        prepared = transaction.prepare_commit()
        resident_result = V1SampledDeviceResidentDTransactionResult(
            transaction_request_hash=request.request_hash,
            session_design_hash=session.design.design_hash,
            hbm_forecast_hash=session.hbm_forecast.forecast_hash,
            registered_look_index=request.registered_look_index,
            look_receipts=tuple(resident_look_receipts),
            provider_receipt_hashes=tuple(resident_provider_proofs),
            prior_result_hash=(None if prior_result is None else prior_result.result_hash),
            cumulative_packet_count_after=len(session.look_receipts),
            cumulative_stratum_counts_after=session.stratum_counts,
            cumulative_look_receipt_hashes=tuple(
                receipt.receipt_hash for receipt in session.look_receipts
            ),
            live_session_custody_hash=content_hash(
                {
                    "session_design_hash": session.design.design_hash,
                    "hbm_forecast_hash": session.hbm_forecast.forecast_hash,
                },
                prefix="partition-free-v1-sampled-device-resident-D-live-custody-",
            ),
        )
        dual_receipt = V1SampledDualChartRegisteredLookReceipt(
            transaction_request_hash=request.request_hash,
            packet_plan_hashes=tuple(plan.plan_hash for plan in request.packet_plans),
            provider_receipt_hashes=tuple(resident_provider_proofs),
            corrected_evidence_child_receipt_hashes=tuple(
                corrected_resident_hashes
            ),
            literal_jacobian_child_receipt_hashes=tuple(literal_resident_hashes),
            raw_dual_stream_receipt_hashes=tuple(raw_dual_hashes),
            row_object_separation_receipt_hashes=tuple(separation_hashes),
            corrected_evidence_stream_identity_hash=(
                first_plan.corrected_evidence_stream_identity_hash
            ),
            literal_jacobian_stream_identity_hash=(
                first_plan.literal_jacobian_stream_identity_hash
            ),
            packet_count=len(request.packet_plans),
            capture_batch_count=len(request.packet_plans),
            vjp_count=len(request.packet_plans),
        )
        row_receipt = V1SampledDualChartRowCustodyRegisteredLookReceipt(
            transaction_request_hash=request.request_hash,
            registered_packet_plan_hashes=tuple(
                plan.plan_hash for plan in request.packet_plans
            ),
            dual_chart_receipt_hash=dual_receipt.receipt_hash,
            group_plan_hash=group_plan.group_plan_hash,
            group_commit_receipt=prepared.receipt,
            provider_receipt_hashes=tuple(custody_provider_proofs),
            corrected_reducer_closure_hashes=tuple(corrected_resident_hashes),
            literal_j_reducer_closure_hashes=tuple(literal_resident_hashes),
            corrected_custody_branch_receipt_hashes=tuple(
                corrected_custody_hashes
            ),
            literal_j_custody_branch_receipt_hashes=tuple(literal_custody_hashes),
            corrected_evidence_stream_identity_hash=(
                first_plan.corrected_evidence_stream_identity_hash
            ),
            literal_jacobian_stream_identity_hash=(
                first_plan.literal_jacobian_stream_identity_hash
            ),
            packet_count=len(request.packet_plans),
        )
        boundary = V1SampledV7DeviceResidentDRowLookBoundary(
            resident_result=resident_result,
            dual_chart_receipt=dual_receipt,
            row_prefix_manifest_hash=prepared.receipt.manifest_hash,
            row_packet_look_hashes=prepared.receipt.packet_look_hashes,
            row_role_payload_sha256_pairs=tuple(
                (
                    look.corrected_a_shard.content_sha256,
                    look.literal_j_shard.content_sha256,
                )
                for look in prepared.attachment.manifest.looks[
                    -len(request.packet_plans) :
                ]
            ),
            source_capture_identity_hashes=tuple(
                packet.source_capture_identity_hash
                for packet in group_plan.packet_plans
            ),
            prior_boundary_hash=(
                None if prior_boundary is None else prior_boundary.boundary_hash
            ),
            provider_vjp_count=len(request.packet_plans),
        )
        diagnostics = {
            "provider_vjp_call_count": len(request.packet_plans),
            "provider_vjp_seconds": provider_seconds,
            "intermediate_moment_canonicalization_count": 0,
            "intermediate_fd_materialization_count": 0,
            "route_scale_moment_d2h_bytes": 0,
            "row_custody_host_transfer_bytes": 0,
            "transaction_seconds": time.perf_counter() - started,
        }
        # The exact custody transfer is derivable without trusting provider
        # telemetry: two fp64 role banks for each observed context.
        diagnostics["row_custody_host_transfer_bytes"] = sum(
            2
            * len(plan.fd_request.context_ids)
            * len(plan.reduced_request.route_ids)
            * 8
            for plan in request.packet_plans
        )
        commit = V1SampledV7DeviceResidentDRowLookCommit(
            boundary=boundary,
            row_custody_receipt=row_receipt,
            row_prefix_attachment=prepared.attachment,
            runtime_diagnostics=diagnostics,
        )
        published = transaction.publish_prepared(
            expected_receipt_hash=prepared.receipt.receipt_hash
        )
        commit.acknowledge_head_publication(published.attachment)
        return commit
    except BaseException:
        transaction.abort()
        session.evict()
        raise


__all__ = [
    "SAMPLED_V7_DEVICE_RESIDENT_D_ROW_AUTHORITY",
    "SAMPLED_V7_DEVICE_RESIDENT_D_ROW_BOUNDARY_SCHEMA",
    "V1SampledV7DeviceResidentDRowLookBoundary",
    "V1SampledV7DeviceResidentDRowLookCommit",
    "derive_v1_sampled_v7_original_capture_binding",
    "execute_v1_sampled_v7_device_resident_d_row_look",
]
