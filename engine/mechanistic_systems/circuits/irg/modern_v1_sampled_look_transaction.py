"""Atomic registered-look execution for sampled modern-v1 discovery.

The provider tee is intentionally a physical transport primitive, not a
transaction manager: one child can have consumed a packet before another
child refuses it.  This module supplies the missing authority boundary.  It
restores the last immutable FD boundary into private ephemeral state, executes
one pre-registered packet for every sampling stratum, verifies the provider
and both typed child closures, and only then returns a new immutable boundary.

No in-flight accumulator or partial reduced packet is exposed.  Consequently
a provider exception, child exception, receipt mismatch, or process crash
before the returned commit is durably published leaves the prior snapshot as
the only canonical continuation state.  Schema v5 also splits preparation
from row-HEAD publication so the outer commit can be durably bound before the
single irreversible custody publication.

This remains proposal-discovery evidence.  A successful look cannot certify
F/Q, open the paired BCE oracle, or commit a mask.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import time
from types import MappingProxyType
from typing import Any, Callable, Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_proposal_packet_gpu import (
    V1GPUReducedProposalPacket,
    _copy_aggregate_to_host_array,
)
from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    V1StratifiedCorrectedAttributionMoments,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_proposal_streaming import (
    CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS,
    LITERAL_GATE_JACOBIAN_ROW_SEMANTICS,
    RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
    SAMPLED_DISCOVERY_AUTHORITY_SCOPE,
    SAMPLED_FD_STREAM_REQUEST_SCHEMA,
    SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2,
    SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA,
    SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2,
    SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA_V2,
    V1SampledFDAppendResult,
    V1SampledFDLossOnlyAdapter,
    V1SampledFDStreamRequest,
    V1SampledProposalDiscoveryClosure,
    V1SampledReducedPacketLossOnlyAdapter,
    V1SampledReducedPacketStreamRequest,
    V1SampledReducedStreamReceipt,
    _SampledStreamTransaction,
    _canonical_device,
    _device_identity,
    _identity_axis,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDLookReceipt,
    V1SampledDeviceResidentDSession,
)
from mechanistic_systems.circuits.irg.modern_v1_stratified_fd_gpu import (
    V1StratifiedDeviceFDTrustAccumulator,
    V1StratifiedFDAccumulatorSnapshot,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowChunk,
    V1SampledDualChartRowGroupCommitReceipt,
    V1SampledDualChartRowGroupPlan,
    V1SampledDualChartRowGroupTransaction,
    V1SampledDualChartRowLivePrefixAttachment,
    V1SampledDualChartRowPrefixManifest,
    begin_v1_sampled_dual_chart_row_group_transaction,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.axis_identity import ordered_string_axis_hash

try:  # pragma: no cover - exercised by import behavior, not local logic.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
from mechanistic_systems.measurement.optimized_irg_providers import (
    _RouteChunkedLossOnlyConsumer,
    _RouteChunkedLossOnlyTee,
    build_route_chunked_dual_chart_fanout,
)


SAMPLED_LOOK_PACKET_PLAN_SCHEMA: Final = "partition_free_v1_sampled_look_packet_plan_v1"
SAMPLED_LOOK_PACKET_PLAN_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_dual_chart_look_packet_plan_v2"
)
SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_registered_look_transaction_request_v1"
)
SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_registered_look_transaction_commit_v1"
)
SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V1: Final = (
    "restore_private_execute_all_strata_verify_then_publish_v1"
)
SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA: Final = (
    "partition_free_v1_sampled_registered_look_transaction_request_v2"
)
SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA: Final = (
    "partition_free_v1_sampled_registered_look_transaction_commit_v2"
)
SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND: Final = (
    "restore_private_execute_all_registered_transport_chunks_verify_then_publish_v2"
)
SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3: Final = (
    "partition_free_v1_sampled_registered_look_bounded_memory_transaction_request_v3"
)
SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V3: Final = (
    "partition_free_v1_sampled_registered_look_bounded_memory_transaction_commit_v3"
)
SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V3: Final = (
    "restore_private_incrementally_merge_verified_packets_then_publish_hash_ledger_v3"
)
SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4: Final = (
    "partition_free_v1_sampled_dual_chart_registered_look_transaction_request_v4"
)
SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V4: Final = (
    "partition_free_v1_sampled_dual_chart_registered_look_transaction_commit_v4"
)
SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V4: Final = (
    "restore_private_dual_chart_same_vjp_merge_verified_packets_then_publish_v4"
)
SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5: Final = (
    "partition_free_v1_sampled_dual_chart_row_custody_transaction_request_v5"
)
SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V5: Final = (
    "partition_free_v1_sampled_dual_chart_row_custody_transaction_commit_v5"
)
SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V5: Final = (
    "restore_private_dual_chart_stage_rows_close_reducers_then_publish_once_v5"
)
SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6: Final = (
    "partition_free_v1_sampled_device_resident_D_row_custody_transaction_request_v6"
)
SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V6: Final = (
    "device_resident_D_tee_same_vjp_into_statistics_and_exact_rows_publish_once_v6"
)
SAMPLED_DUAL_CHART_REGISTERED_LOOK_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_registered_look_receipt_v1"
)
SAMPLED_DUAL_CHART_ROW_CUSTODY_REGISTERED_LOOK_RECEIPT_SCHEMA_V5: Final = (
    "partition_free_v1_sampled_dual_chart_row_custody_registered_look_receipt_v5"
)
SAMPLED_LOOK_AGGREGATE_SCORE_CONTRIBUTION_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_look_aggregate_score_contribution_v1"
)
SAMPLED_LOOK_AGGREGATE_SCORE_RESULT_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_look_aggregate_score_result_v1"
)
SAMPLED_DEVICE_RESIDENT_D_TRANSACTION_RESULT_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_device_resident_D_transaction_result_v1"
)

_PLAN_PREFIX: Final = "partition-free-v1-sampled-look-packet-plan-"
_REQUEST_PREFIX: Final = "partition-free-v1-sampled-look-transaction-request-"
_COMMIT_PREFIX: Final = "partition-free-v1-sampled-look-transaction-commit-"
_PROVIDER_PREFIX: Final = "partition-free-v1-sampled-look-provider-receipt-"
_DUAL_RECEIPT_PREFIX: Final = "partition-free-v1-sampled-look-dual-chart-receipt-"
_DUAL_ROW_CUSTODY_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-look-dual-row-custody-receipt-"
)
_PROVIDER_CONTEXT_PREFIX: Final = "irg-device-panel-contexts-"
_FD_CHILD_LEDGER_PREFIX: Final = "partition-free-v1-sampled-fd-child-ledger-"
_FD_SOURCE_AXIS_PREFIX: Final = "partition-free-v1-sampled-source-axis-"
_DUAL_ROW_GROUP_BRANCH_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-dual-row-group-branch-receipt-"
)
_LOOK_SCORE_CONTRIBUTION_PREFIX: Final = (
    "partition-free-v1-sampled-look-score-aggregate-contribution-"
)
_LOOK_SCORE_RESULT_PREFIX: Final = (
    "partition-free-v1-sampled-look-score-aggregate-result-"
)
_DEVICE_RESIDENT_D_BRANCH_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-device-resident-D-branch-"
)
_DEVICE_RESIDENT_D_TRANSACTION_RESULT_PREFIX: Final = (
    "partition-free-v1-sampled-device-resident-D-transaction-result-"
)
_DEVICE_RESIDENT_D_CHUNK_SOURCE_PREFIX: Final = (
    "partition-free-v1-sampled-device-resident-D-same-capture-chunk-"
)
_LE_F64: Final = np.dtype("<f8")
_PREPARED_V5_ISSUANCE_TOKEN: Final = object()
_V5_PROVIDER_NORMALIZED_PROOF_PREFIX: Final = (
    "partition-free-v1-sampled-v5-provider-normalized-proof-"
)
SAMPLED_DUAL_CHART_CUSTODY_PROVIDER_VERIFICATION_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_dual_chart_custody_only_provider_"
    "verification_v1"
)
_CUSTODY_PROVIDER_VERIFICATION_PREFIX: Final = (
    "partition-free-v1-sampled-dual-chart-custody-provider-verification-"
)
_CUSTODY_PROVIDER_VERIFICATION_ISSUER_TOKEN: Final = object()


def _commit_schema_for_request_schema(request_schema: str) -> str:
    if request_schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V1:
        return SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V1
    if request_schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA:
        return SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA
    if request_schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3:
        return SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V3
    if request_schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4:
        return SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V4
    if request_schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5:
        return SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V5
    raise ValueError("sampled look transaction-request schema is invalid")


def _device_resident_row_plan_scientific_hash(
    plan: V1SampledDualChartRowGroupPlan,
) -> str:
    """Hash scientific row-custody coordinates, excluding resource telemetry.

    The group scratch forecast is still integrity-checked and enforced before
    execution, but free-space telemetry is an operational refusal coordinate,
    not part of restart identity.  V6 therefore binds this stable plan view in
    its request hash while retaining exact in-process custody of the full plan.
    """

    if type(plan) is not V1SampledDualChartRowGroupPlan:
        raise TypeError("device-resident row plan must have its exact type")
    plan.assert_integrity()
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_device_resident_D_row_plan_science_v1",
            "registered_look_index": plan.registered_look_index,
            # The exact predecessor manifest remains mandatory live custody in
            # ``plan`` and is checked against the prior attachment before any
            # provider work.  It is deliberately absent from restart science
            # identity: the manifest also binds the pre-outcome filesystem
            # free-space forecast, which can lawfully change after a crash.
            # V6's stable predecessor is instead the prior resident-result
            # hash in the enclosing request plus the V7 exact-row content
            # chain checked by the trajectory controller.
            "first_prefix_look_ordinal": plan.first_prefix_look_ordinal,
            "registered_d_prefix_hash": plan.registered_d_prefix_hash,
            "chart_identity_hash": plan.chart_identity_hash,
            "current_mask_hash": plan.current_mask_hash,
            "proposal_generation": plan.proposal_generation,
            "route_axis_hash": plan.route_axis_hash,
            "route_id_axis_hash": plan.route_id_axis_hash,
            "route_count": plan.route_count,
            "corrected_a_stream_identity_hash": (
                plan.corrected_a_stream_identity_hash
            ),
            "literal_j_stream_identity_hash": plan.literal_j_stream_identity_hash,
            "packet_plan_hashes": tuple(
                packet.packet_plan_hash for packet in plan.packet_plans
            ),
            "registered_route_chunk_size": plan.registered_route_chunk_size,
            "raw_shards_carry_normalized_weights": False,
        },
        prefix="partition-free-v1-sampled-device-resident-D-row-plan-science-",
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _provider_plan_identity_hash(value: object) -> str:
    """Resolve one preregistered measurement-plan identity without outcomes."""

    if type(value) is str:
        return _identity(value, name="provider measurement-plan identity hash")
    candidates = tuple(
        getattr(value, name, None)
        for name in (
            "request_hash",
            "measurement_request_hash",
            "plan_hash",
            "identity_hash",
            # ``MeasurementRequest`` owns this canonical predictable identity.
            # It is intentionally last so an adapter that also exposes one of
            # the historical aliases must agree rather than silently winning.
            "request_id",
        )
    )
    identities = tuple(
        candidate
        for candidate in candidates
        if type(candidate) is str and candidate
    )
    if len(set(identities)) != 1:
        raise TypeError(
            "v5 measurement request must expose one unambiguous preregistered "
            "provider-plan identity"
        )
    return identities[0]


def _source_capture_identity_hash(value: object) -> str:
    """Resolve the preregistered capture-plan identity from one V5 request.

    A production :class:`MeasurementRequest` carries this identity in its
    predictable ``payload``; narrow adapters may expose the same value as a
    direct attribute.  If both surfaces exist, exact agreement is required.
    The identity is pre-outcome: it names the requested capture/VJP source,
    not a digest learned from emitted rows.
    """

    candidates: list[object] = [
        getattr(value, "source_capture_identity_hash", None),
    ]
    payload = getattr(value, "payload", None)
    if isinstance(payload, Mapping):
        candidates.append(payload.get("source_capture_identity_hash"))
    identities = tuple(
        candidate
        for candidate in candidates
        if type(candidate) is str and candidate
    )
    if len(set(identities)) != 1:
        raise TypeError(
            "v5 measurement request must expose one unambiguous preregistered "
            "source-capture identity"
        )
    return identities[0]


_V5_SENSITIVE_PROVIDER_CLAIM_FRAGMENTS: Final = (
    "authority",
    "directory",
    "oracle",
    "pareto",
    "path",
    "postselect",
    "post_selection",
    "terminal_validation",
    "uri",
)


def _refuse_unregistered_v5_provider_claims(
    receipt: Mapping[str, object],
    *,
    scope: str,
) -> None:
    """Refuse provider-owned path, postselection, or authority assertions.

    V5 hashes a closed scientific projection rather than the provider's open
    telemetry mapping.  Ordinary timing/device telemetry may therefore remain
    extensible, while semantically privileged claims cannot hide among those
    unregistered fields.  Typed reducer/custody children are verified through
    their own exact hashes and are deliberately outside this provider-owned
    claim surface.
    """

    for raw_name in receipt:
        if type(raw_name) is not str:
            raise TypeError(f"{scope} receipt keys must be strings")
        name = raw_name.casefold()
        if any(
            fragment in name
            for fragment in _V5_SENSITIVE_PROVIDER_CLAIM_FRAGMENTS
        ):
            raise ValueError(
                f"{scope} receipt contains unregistered privileged claim "
                f"{raw_name!r}"
            )


def _positive_int(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be one positive integer")
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be one nonnegative integer")
    return int(value)


def _accumulate_runtime_diagnostics(
    destination: dict[str, object],
    source: Mapping[str, object],
) -> None:
    """Merge small scalar physical metrics without entering scientific state."""

    for name, raw_value in source.items():
        if isinstance(raw_value, bool) or not isinstance(
            raw_value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"sampled runtime diagnostic {name} is not scalar")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"sampled runtime diagnostic {name} is invalid")
        if name.endswith("_count") or name.endswith("_bytes"):
            if not value.is_integer():
                raise ValueError(f"sampled runtime diagnostic {name} is not integral")
            destination[name] = int(destination.get(name, 0)) + int(value)
        else:
            destination[name] = float(destination.get(name, 0.0)) + value


def _fd_snapshot_array_footprint(
    snapshot: V1StratifiedFDAccumulatorSnapshot,
) -> tuple[int, int]:
    """Return O(strata) array count/bytes without scanning route values."""

    count = 0
    byte_count = 0
    for _name, child in snapshot.stratum_snapshots:
        for value in (
            child.active_buffer_rows,
            child.mean,
            child.weighted_second_moment_diagonal,
        ):
            if value is None:
                continue
            array = np.asarray(value)
            count += 1
            byte_count += int(array.nbytes)
    return count, byte_count


def _array_custody(
    values: Sequence[np.ndarray | None],
) -> tuple[object, ...]:
    """Capture exact readonly array objects and their O(1) layout metadata."""

    arrays = tuple(values)
    metadata = tuple(
        (
            None
            if value is None
            else (
                value.shape,
                value.dtype.str,
                value.strides,
                value.flags.c_contiguous,
            )
        )
        for value in arrays
    )
    return arrays, metadata


def _retains_array_custody(
    values: Sequence[np.ndarray | None],
    custody: tuple[object, ...],
) -> bool:
    arrays, metadata = custody
    current = tuple(values)
    if len(current) != len(arrays) or any(
        value is not expected for value, expected in zip(current, arrays, strict=True)
    ):
        return False
    return all(
        value is None or not value.flags.writeable for value in current
    ) and tuple(
        (
            None
            if value is None
            else (
                value.shape,
                value.dtype.str,
                value.strides,
                value.flags.c_contiguous,
            )
        )
        for value in current
    ) == metadata


def _capture_moments_custody(
    moments: V1StratifiedCorrectedAttributionMoments,
) -> tuple[object, ...]:
    scalar_state = (
        moments.campaign_contract_hash,
        moments.model_identity_hash,
        moments.query_contract_hash,
        moments.preprocessing_hash,
        moments.current_mask_hash,
        moments.raw_logit_stream_source_hash,
        moments.sampling_law_hash,
        moments.proposal_generation,
        moments.route_count,
        moments.route_axis_hash,
        moments.stratum_names,
        moments.population_stratum_counts,
        moments.sealed_population_order_hashes,
        moments.stratum_counts,
        moments.observed_context_ids_by_stratum,
        moments.observed_draw_ids_by_stratum,
        moments.look_index,
        moments.prior_state_hash,
        moments.schema,
    )
    arrays = _array_custody(
        (
            moments.signed_mean_by_stratum,
            moments.signed_m2_by_stratum,
            moments.absolute_mean_by_stratum,
            moments.absolute_m2_by_stratum,
        )
    )
    return scalar_state, arrays, moments._state_hash


def _retains_moments_custody(
    moments: V1StratifiedCorrectedAttributionMoments,
    custody: tuple[object, ...],
    expected_hash: str,
) -> bool:
    scalar_state, arrays, cached_hash = custody
    return (
        (
            moments.campaign_contract_hash,
            moments.model_identity_hash,
            moments.query_contract_hash,
            moments.preprocessing_hash,
            moments.current_mask_hash,
            moments.raw_logit_stream_source_hash,
            moments.sampling_law_hash,
            moments.proposal_generation,
            moments.route_count,
            moments.route_axis_hash,
            moments.stratum_names,
            moments.population_stratum_counts,
            moments.sealed_population_order_hashes,
            moments.stratum_counts,
            moments.observed_context_ids_by_stratum,
            moments.observed_draw_ids_by_stratum,
            moments.look_index,
            moments.prior_state_hash,
            moments.schema,
        )
        == scalar_state
        and _retains_array_custody(
            (
                moments.signed_mean_by_stratum,
                moments.signed_m2_by_stratum,
                moments.absolute_mean_by_stratum,
                moments.absolute_m2_by_stratum,
            ),
            arrays,
        )
        and moments._state_hash == cached_hash == expected_hash
    )


def _fd_child_scalar_state(snapshot: Any) -> tuple[object, ...]:
    return (
        snapshot.schema,
        snapshot.producer_algorithm_id,
        snapshot.producer_implementation,
        snapshot.route_ids,
        snapshot.ordered_route_axis_hash,
        snapshot.sketch_rank,
        snapshot.dtype_identity,
        snapshot.filled_count,
        snapshot.cumulative_shrinkage,
        snapshot.total_weight,
        snapshot.context_count,
        snapshot.context_ids,
        snapshot.latest_source_row_hashes,
        snapshot.snapshot_hash,
    )


def _capture_fd_snapshot_custody(
    snapshot: V1StratifiedFDAccumulatorSnapshot,
) -> tuple[object, ...]:
    parent_state = (
        snapshot.design_hash,
        snapshot.design_stratum_names,
        snapshot.route_axis_hash,
        snapshot.sketch_rank,
        snapshot.packet_boundary_index,
        snapshot.source_measurement_hashes_by_stratum,
        snapshot.schema,
    )
    children = tuple(
        (
            name,
            child,
            _fd_child_scalar_state(child),
            child.bounded_source_proof,
            _array_custody(
                (
                    child.active_buffer_rows,
                    child.mean,
                    child.weighted_second_moment_diagonal,
                )
            ),
        )
        for name, child in snapshot.stratum_snapshots
    )
    return parent_state, snapshot.stratum_snapshots, children, snapshot._snapshot_hash


def _retains_fd_snapshot_custody(
    snapshot: V1StratifiedFDAccumulatorSnapshot,
    custody: tuple[object, ...],
    expected_hash: str,
) -> bool:
    parent_state, stratum_snapshots, children, cached_hash = custody
    if (
        (
            snapshot.design_hash,
            snapshot.design_stratum_names,
            snapshot.route_axis_hash,
            snapshot.sketch_rank,
            snapshot.packet_boundary_index,
            snapshot.source_measurement_hashes_by_stratum,
            snapshot.schema,
        )
        != parent_state
        or snapshot.stratum_snapshots is not stratum_snapshots
        or len(snapshot.stratum_snapshots) != len(children)
        or snapshot._snapshot_hash != cached_hash
        or cached_hash != expected_hash
    ):
        return False
    for (name, child), child_custody in zip(
        snapshot.stratum_snapshots,
        children,
        strict=True,
    ):
        (
            expected_name,
            expected_child,
            scalar_state,
            bounded_source_proof,
            arrays,
        ) = child_custody
        if (
            name != expected_name
            or child is not expected_child
            or _fd_child_scalar_state(child) != scalar_state
            or child.bounded_source_proof is not bounded_source_proof
            or not _retains_array_custody(
                (
                    child.active_buffer_rows,
                    child.mean,
                    child.weighted_second_moment_diagonal,
                ),
                arrays,
            )
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class V1SampledLookPacketPlan:
    """One same-stratum physical acquisition inside a registered look."""

    reduced_request: V1SampledReducedPacketStreamRequest
    fd_request: V1SampledFDStreamRequest
    corrected_evidence_stream_identity_hash: str | None = None
    literal_jacobian_stream_identity_hash: str | None = None
    same_capture_and_vjp_required: bool = False
    schema: str = SAMPLED_LOOK_PACKET_PLAN_SCHEMA
    _plan_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.reduced_request) is not V1SampledReducedPacketStreamRequest:
            raise TypeError("sampled look plan requires its exact reduced request")
        if type(self.fd_request) is not V1SampledFDStreamRequest:
            raise TypeError("sampled look plan requires its exact FD request")
        self.reduced_request.assert_integrity()
        self.fd_request.assert_integrity()
        packet = self.reduced_request.packet_request
        fd = self.fd_request
        checks = {
            "context ledger": (packet.expected_context_ids, fd.context_ids),
            "stratum ledger": (packet.expected_stratum_labels, fd.stratum_labels),
            "route axis": (self.reduced_request.route_ids, fd.design.route_ids),
            "destination transport": (
                self.reduced_request.destination_transport_hash,
                fd.destination_transport_hash,
            ),
            "declared device": (
                self.reduced_request.declared_device,
                fd.declared_device,
            ),
            "registered route chunk": (
                packet.registered_route_chunk_size,
                fd.registered_route_chunk_size,
            ),
            "packet index": (packet.packet_index, fd.packet_index),
        }
        for name, (left, right) in checks.items():
            if left != right:
                raise ValueError(f"sampled look plan {name} differs by child")
        if self.schema == SAMPLED_LOOK_PACKET_PLAN_SCHEMA:
            if (
                self.corrected_evidence_stream_identity_hash is not None
                or self.literal_jacobian_stream_identity_hash is not None
                or self.same_capture_and_vjp_required is not False
                or self.reduced_request.schema
                != SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA
                or fd.schema != SAMPLED_FD_STREAM_REQUEST_SCHEMA
            ):
                raise ValueError(
                    "legacy sampled look plans cannot claim dual-chart authority"
                )
            if (
                self.reduced_request.raw_logit_stream_identity_hash
                != fd.raw_logit_stream_identity_hash
                or packet.raw_logit_stream_source_hash
                != fd.design.raw_logit_stream_source_hash
            ):
                raise ValueError("sampled look child source-stream identities differ")
        elif self.schema == SAMPLED_LOOK_PACKET_PLAN_SCHEMA_V2:
            evidence = _identity(
                self.corrected_evidence_stream_identity_hash,
                name="corrected-evidence stream identity hash",
            )
            jacobian = _identity(
                self.literal_jacobian_stream_identity_hash,
                name="literal-Jacobian stream identity hash",
            )
            if (
                evidence == jacobian
                or self.same_capture_and_vjp_required is not True
                or self.reduced_request.schema
                != SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2
                or fd.schema != SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2
                or self.reduced_request.raw_target_logit_row_semantics
                != CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
                or fd.raw_target_logit_row_semantics
                != LITERAL_GATE_JACOBIAN_ROW_SEMANTICS
                or self.reduced_request.raw_logit_stream_identity_hash != evidence
                or packet.raw_logit_stream_source_hash != evidence
                or fd.raw_logit_stream_identity_hash != jacobian
                or fd.design.raw_logit_stream_source_hash != jacobian
            ):
                raise ValueError(
                    "sampled dual-chart plan collapsed, swapped, or mistyped its "
                    "evidence/Jacobian streams"
                )
            object.__setattr__(
                self,
                "corrected_evidence_stream_identity_hash",
                evidence,
            )
            object.__setattr__(
                self,
                "literal_jacobian_stream_identity_hash",
                jacobian,
            )
        else:
            raise ValueError("sampled look packet-plan schema is invalid")
        if len(set(fd.stratum_labels)) != 1:
            raise ValueError("sampled look packet must contain one stratum")

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "reduced_request_hash": self.reduced_request.request_hash,
            "fd_request_hash": self.fd_request.request_hash,
        }
        if self.schema == SAMPLED_LOOK_PACKET_PLAN_SCHEMA_V2:
            payload.update(
                {
                    "corrected_evidence_stream_identity_hash": (
                        self.corrected_evidence_stream_identity_hash
                    ),
                    "literal_jacobian_stream_identity_hash": (
                        self.literal_jacobian_stream_identity_hash
                    ),
                    "same_capture_and_vjp_required": (
                        self.same_capture_and_vjp_required
                    ),
                }
            )
        return payload

    @property
    def plan_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_PLAN_PREFIX)
        if self._plan_hash is not None and observed != self._plan_hash:
            raise ValueError("sampled look packet plan was tampered")
        object.__setattr__(self, "_plan_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledRegisteredLookTransactionRequest:
    """Pre-outcome atomic plan for one or more sealed-prefix packets.

    Schema v2 separates the registered statistical look from its physical
    transport packet size.  Packet plans are grouped in the frozen stratum
    order, but a stratum may span several contiguous packets.  A stratum may
    contribute zero packets only after its finite population is already a
    census; at least one other stratum must advance.  None of those packets is
    independently publishable.  Schema v1 remains readable and constructible
    for exact one-packet-per-stratum replay.
    """

    packet_plans: tuple[V1SampledLookPacketPlan, ...]
    sketch_rank: int
    registered_sequential_design_hash: str
    registered_look_index: int
    registered_look_cumulative_stratum_counts: tuple[int, ...]
    prior_corrected_attribution_state: (
        V1StratifiedCorrectedAttributionMoments | None
    ) = None
    prior_fd_snapshot: V1StratifiedFDAccumulatorSnapshot | None = None
    prior_device_resident_result: (
        V1SampledDeviceResidentDTransactionResult | None
    ) = None
    dual_chart_row_group_plan: V1SampledDualChartRowGroupPlan | None = None
    transaction_kind: str = SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND
    schema: str = SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA
    _request_hash: str | None = field(init=False, default=None, repr=False)
    _validated_prior_corrected_attribution_state: (
        V1StratifiedCorrectedAttributionMoments | None
    ) = field(init=False, default=None, repr=False, compare=False)
    _validated_prior_fd_snapshot: V1StratifiedFDAccumulatorSnapshot | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _validated_prior_corrected_attribution_state_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _validated_prior_corrected_attribution_custody: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _validated_prior_fd_snapshot_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _validated_prior_fd_snapshot_custody: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _validated_prior_device_resident_result: (
        V1SampledDeviceResidentDTransactionResult | None
    ) = field(init=False, default=None, repr=False, compare=False)
    _validated_prior_device_resident_result_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _request_hash_custody: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        plans = tuple(self.packet_plans)
        if not plans:
            raise ValueError(
                "one sampled registered look requires at least one packet total"
            )
        if any(type(plan) is not V1SampledLookPacketPlan for plan in plans):
            raise TypeError("sampled look transaction requires exact packet plans")
        for plan in plans:
            plan.plan_hash
        rank = _positive_int(self.sketch_rank, name="sampled look sketch rank")
        design_hash = _identity(
            self.registered_sequential_design_hash,
            name="registered sequential design hash",
        )
        look_index = _positive_int(
            self.registered_look_index,
            name="registered look index",
        )
        design = plans[0].fd_request.design
        if rank > len(design.route_ids):
            raise ValueError("sampled look sketch rank exceeds the route axis")
        observed_strata = tuple(plan.fd_request.stratum_labels[0] for plan in plans)
        if self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V1:
            if self.transaction_kind != SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V1:
                raise ValueError("legacy sampled look transaction kind is invalid")
            if len(plans) != len(design.stratum_names):
                raise ValueError(
                    "legacy registered looks require one packet per stratum"
                )
            expected_strata = design.stratum_names
        elif self.schema in (
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6,
        ):
            expected_transaction_kind = (
                SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V6
                if self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6
                else (
                    SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V5
                    if self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5
                    else (
                        SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V4
                        if self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4
                        else (
                            SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V3
                            if self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3
                            else SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND
                        )
                    )
                )
            )
            if self.transaction_kind != expected_transaction_kind:
                raise ValueError("sampled look transaction kind is invalid")
            packet_counts_by_stratum = tuple(
                observed_strata.count(name) for name in design.stratum_names
            )
            expected_strata = tuple(
                name
                for name, count in zip(
                    design.stratum_names,
                    packet_counts_by_stratum,
                    strict=True,
                )
                for _ in range(count)
            )
        else:
            raise ValueError("sampled look transaction-request schema is invalid")
        dual_chart = self.schema in (
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6,
        )
        row_custody = self.schema in (
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6,
        )
        expected_plan_schema = (
            SAMPLED_LOOK_PACKET_PLAN_SCHEMA_V2
            if dual_chart
            else SAMPLED_LOOK_PACKET_PLAN_SCHEMA
        )
        if any(plan.schema != expected_plan_schema for plan in plans):
            raise ValueError(
                "sampled look transaction mixed legacy and dual-chart packet plans"
            )
        if observed_strata != expected_strata:
            raise ValueError(
                "sampled look packet groups must follow the frozen stratum order "
                "exactly"
            )

        group_plan = self.dual_chart_row_group_plan
        if row_custody:
            if type(group_plan) is not V1SampledDualChartRowGroupPlan:
                raise TypeError(
                    "sampled v5 transaction requires its exact dual-row group plan"
                )
            group_plan.assert_integrity()
            first_packet = plans[0].reduced_request.packet_request
            exact_group_coordinates = {
                "registered look": (
                    group_plan.registered_look_index,
                    look_index,
                ),
                "mask": (group_plan.current_mask_hash, first_packet.current_mask_hash),
                "generation": (
                    group_plan.proposal_generation,
                    first_packet.proposal_generation,
                ),
                "route count": (group_plan.route_count, first_packet.route_count),
                # The reduced packet stream is indexed only by the ordered
                # constructive route-ID transport.  The row-custody group also
                # carries the distinct physical topology/block axis, which is
                # joined later to the compact route and native-fiber owners.
                # Comparing the packet hash to the physical hash collapses the
                # two axes and makes that later join unsatisfiable.
                "packet ordered route axis": (
                    group_plan.route_id_axis_hash,
                    first_packet.route_axis_hash,
                ),
                "route-ID axis": (
                    group_plan.route_id_axis_hash,
                    ordered_string_axis_hash(
                        plans[0].reduced_request.route_ids,
                        name="sampled v5 row-custody route-ID axis",
                    ),
                ),
                "corrected-A stream": (
                    group_plan.corrected_a_stream_identity_hash,
                    plans[0].corrected_evidence_stream_identity_hash,
                ),
                "literal-J stream": (
                    group_plan.literal_j_stream_identity_hash,
                    plans[0].literal_jacobian_stream_identity_hash,
                ),
                "route chunk": (
                    group_plan.registered_route_chunk_size,
                    first_packet.registered_route_chunk_size,
                ),
            }
            for name, (left, right) in exact_group_coordinates.items():
                if left != right:
                    raise ValueError(
                        f"sampled v5 row-custody group differs on {name}"
                    )
            group_packets = tuple(group_plan.packet_plans)
            if len(group_packets) != len(plans):
                raise ValueError(
                    "sampled v5 row-custody group packet ledger is partial"
                )
            for ordinal, (plan, group_packet) in enumerate(
                zip(plans, group_packets, strict=True)
            ):
                packet = plan.reduced_request.packet_request
                expected_reducer_plans = (
                    plan.reduced_request.request_hash,
                    plan.fd_request.request_hash,
                )
                if (
                    group_packet.packet_ordinal != ordinal
                    or group_packet.registered_packet_plan_hash != plan.plan_hash
                    or group_packet.reducer_plan_identity_hashes
                    != expected_reducer_plans
                    or group_packet.context_ids != plan.fd_request.context_ids
                    or group_packet.draw_ids != packet.expected_draw_ids
                    or group_packet.stratum_labels != plan.fd_request.stratum_labels
                ):
                    raise ValueError(
                        "sampled v5 group packet differs from its exact registered "
                        "packet/reducer plan"
                    )
        elif group_plan is not None:
            raise ValueError(
                "sampled v1-v4 transactions cannot claim dual-row group custody"
            )

        live_prior = self.prior_device_resident_result
        if self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6:
            if (
                self.prior_fd_snapshot is not None
                or self.prior_corrected_attribution_state is not None
            ):
                raise ValueError(
                    "sampled V6 device-resident row custody cannot carry portable prior state"
                )
        if live_prior is not None and (
            self.prior_fd_snapshot is not None
            or self.prior_corrected_attribution_state is not None
        ):
            raise ValueError(
                "sampled look cannot mix portable and device-resident prior state"
            )
        validated_live_prior_hash: str | None = None
        if live_prior is not None:
            if type(live_prior) is not V1SampledDeviceResidentDTransactionResult:
                raise TypeError(
                    "device-resident continuation requires its exact prior result"
                )
            validated_live_prior_hash = live_prior.result_hash
            if look_index != live_prior.registered_look_index + 1:
                raise ValueError(
                    "device-resident sampled looks must advance the prior result by one"
                )
        first_packet_index = (
            live_prior.cumulative_packet_count_after
            if live_prior is not None
            else (
                0
                if self.prior_fd_snapshot is None
                else self.prior_fd_snapshot.packet_boundary_index
            )
        )
        prior_counts = [0 for _ in design.stratum_names]
        validated_prior_snapshot_hash: str | None = None
        if live_prior is not None:
            if len(live_prior.cumulative_stratum_counts_after) != len(
                design.stratum_names
            ):
                raise ValueError(
                    "device-resident prior result has another stratum axis"
                )
            prior_counts = list(live_prior.cumulative_stratum_counts_after)
        elif self.prior_fd_snapshot is not None:
            snapshot = self.prior_fd_snapshot
            if type(snapshot) is not V1StratifiedFDAccumulatorSnapshot:
                raise TypeError("sampled look prior state requires its exact snapshot")
            validated_prior_snapshot_hash = snapshot.assert_integrity()
            if (
                snapshot.design_hash != design.design_hash
                or snapshot.design_stratum_names != design.stratum_names
                or snapshot.route_axis_hash != design.route_axis_hash
                or snapshot.sketch_rank != rank
            ):
                raise ValueError(
                    "sampled look prior FD snapshot differs from its design"
                )
            prior_counts = [
                child.context_count for _name, child in snapshot.stratum_snapshots
            ]
        prior_moments = self.prior_corrected_attribution_state
        validated_prior_moments_hash: str | None = None
        if prior_moments is None:
            declared_prior_counts = tuple(prior_counts)
            if self.prior_fd_snapshot is not None and live_prior is None:
                raise ValueError(
                    "sampled look prior moments and FD snapshot must coexist"
                )
            if look_index != 1 and live_prior is None:
                raise ValueError(
                    "a later sampled look requires portable or device-resident prior state"
                )
        else:
            if type(prior_moments) is not V1StratifiedCorrectedAttributionMoments:
                raise TypeError("sampled look prior moments require their exact type")
            validated_prior_moments_hash = prior_moments.assert_integrity()
            if look_index != prior_moments.look_index + 1:
                raise ValueError(
                    "sampled registered looks must advance the prior moments by one"
                )
            declared_prior_counts = tuple(prior_moments.stratum_counts)
            if self.prior_fd_snapshot is None:
                raise ValueError(
                    "sampled look prior moments and FD snapshot must coexist"
                )
            first_packet = plans[0].reduced_request.packet_request
            moment_coordinates = {
                "campaign": (
                    prior_moments.campaign_contract_hash,
                    first_packet.campaign_contract_hash,
                ),
                "model": (
                    prior_moments.model_identity_hash,
                    first_packet.model_identity_hash,
                ),
                "query": (
                    prior_moments.query_contract_hash,
                    first_packet.query_contract_hash,
                ),
                "preprocessing": (
                    prior_moments.preprocessing_hash,
                    first_packet.preprocessing_hash,
                ),
                "mask": (
                    prior_moments.current_mask_hash,
                    first_packet.current_mask_hash,
                ),
                "raw-logit stream": (
                    prior_moments.raw_logit_stream_source_hash,
                    first_packet.raw_logit_stream_source_hash,
                ),
                "sampling law": (
                    prior_moments.sampling_law_hash,
                    first_packet.sampling_law_hash,
                ),
                "generation": (
                    prior_moments.proposal_generation,
                    first_packet.proposal_generation,
                ),
                "route axis": (
                    prior_moments.route_axis_hash,
                    first_packet.route_axis_hash,
                ),
                "strata": (
                    prior_moments.stratum_names,
                    first_packet.stratum_names,
                ),
                "population counts": (
                    prior_moments.population_stratum_counts,
                    first_packet.population_stratum_counts,
                ),
                "population orders": (
                    prior_moments.sealed_population_order_hashes,
                    first_packet.sealed_population_order_hashes,
                ),
            }
            for name, (left, right) in moment_coordinates.items():
                if left != right:
                    raise ValueError(f"sampled look prior moments disagree on {name}")
        if tuple(prior_counts) != declared_prior_counts:
            raise ValueError(
                "prior corrected-attribution and FD stratum counts diverged"
            )

        shared_packet = plans[0].reduced_request.packet_request
        shared = {
            "design hash": design.design_hash,
            "route axis": plans[0].reduced_request.route_ids,
            "destination transport": (
                plans[0].reduced_request.destination_transport_hash
            ),
            "corrected-evidence stream": (
                plans[0].reduced_request.raw_logit_stream_identity_hash
            ),
            "declared device": plans[0].reduced_request.declared_device,
            "route chunk": shared_packet.registered_route_chunk_size,
            "population counts": shared_packet.population_stratum_counts,
            "population order hashes": shared_packet.sealed_population_order_hashes,
            "packetization plan": shared_packet.packetization_plan_hash,
        }
        if dual_chart:
            shared["literal-Jacobian stream"] = (
                plans[0].fd_request.raw_logit_stream_identity_hash
            )
        expected_starts = list(prior_counts)
        all_contexts: list[str] = []
        all_draws: list[str] = []
        for offset, plan in enumerate(plans):
            fd = plan.fd_request
            packet = plan.reduced_request.packet_request
            observed = {
                "design hash": fd.design.design_hash,
                "route axis": plan.reduced_request.route_ids,
                "destination transport": plan.reduced_request.destination_transport_hash,
                "corrected-evidence stream": (
                    plan.reduced_request.raw_logit_stream_identity_hash
                ),
                "declared device": plan.reduced_request.declared_device,
                "route chunk": packet.registered_route_chunk_size,
                "population counts": packet.population_stratum_counts,
                "population order hashes": packet.sealed_population_order_hashes,
                "packetization plan": packet.packetization_plan_hash,
            }
            if dual_chart:
                observed["literal-Jacobian stream"] = (
                    plan.fd_request.raw_logit_stream_identity_hash
                )
            if observed != shared:
                raise ValueError(
                    "sampled look packet plans do not share one frozen law"
                )
            if packet.packet_index != first_packet_index + offset:
                raise ValueError("sampled look packet indices are not contiguous")
            if tuple(packet.sealed_order_start_indices) != tuple(expected_starts):
                raise ValueError(
                    "sampled look packet does not continue the sealed prefixes"
                )
            stratum_index = design.stratum_names.index(fd.stratum_labels[0])
            expected_starts[stratum_index] += len(fd.context_ids)
            all_contexts.extend(fd.context_ids)
            all_draws.extend(packet.expected_draw_ids)
        if len(set(all_contexts)) != len(all_contexts):
            raise ValueError("sampled look repeats a context across strata")
        if len(set(all_draws)) != len(all_draws):
            raise ValueError("sampled look repeats a draw identity across strata")
        cumulative = tuple(
            _positive_int(value, name="registered cumulative stratum count")
            for value in self.registered_look_cumulative_stratum_counts
        )
        if cumulative != tuple(expected_starts):
            raise ValueError(
                "sampled look packets do not close the registered cumulative counts"
            )
        if self.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V1:
            population_counts = tuple(shared_packet.population_stratum_counts)
            for stratum, packet_count in enumerate(packet_counts_by_stratum):
                if packet_count != 0:
                    continue
                if not (
                    cumulative[stratum]
                    == prior_counts[stratum]
                    == population_counts[stratum]
                ):
                    raise ValueError(
                        "a zero-packet stratum must already equal its registered "
                        "finite-population census"
                    )
        increments = tuple(
            after - before
            for before, after in zip(prior_counts, cumulative, strict=True)
        )
        if any(increment < 0 for increment in increments) or sum(increments) <= 0:
            raise ValueError(
                "one sampled registered look must be a nonempty monotone prefix "
                "increment"
            )
        object.__setattr__(self, "packet_plans", plans)
        object.__setattr__(self, "sketch_rank", rank)
        object.__setattr__(self, "registered_sequential_design_hash", design_hash)
        object.__setattr__(self, "registered_look_index", look_index)
        object.__setattr__(
            self,
            "registered_look_cumulative_stratum_counts",
            cumulative,
        )
        object.__setattr__(
            self,
            "_validated_prior_corrected_attribution_state",
            prior_moments,
        )
        object.__setattr__(self, "_validated_prior_fd_snapshot", self.prior_fd_snapshot)
        object.__setattr__(
            self,
            "_validated_prior_corrected_attribution_state_hash",
            validated_prior_moments_hash,
        )
        object.__setattr__(
            self,
            "_validated_prior_corrected_attribution_custody",
            (
                None
                if prior_moments is None
                else _capture_moments_custody(prior_moments)
            ),
        )
        object.__setattr__(
            self,
            "_validated_prior_fd_snapshot_hash",
            validated_prior_snapshot_hash,
        )
        object.__setattr__(
            self,
            "_validated_prior_fd_snapshot_custody",
            (
                None
                if self.prior_fd_snapshot is None
                else _capture_fd_snapshot_custody(self.prior_fd_snapshot)
            ),
        )
        object.__setattr__(
            self,
            "_validated_prior_device_resident_result",
            live_prior,
        )
        object.__setattr__(
            self,
            "_validated_prior_device_resident_result_hash",
            validated_live_prior_hash,
        )
        observed_request_hash = content_hash(
            self._hash_payload(),
            prefix=_REQUEST_PREFIX,
        )
        object.__setattr__(self, "_request_hash", observed_request_hash)
        object.__setattr__(
            self,
            "_request_hash_custody",
            self._current_hash_custody(observed_request_hash),
        )

    @property
    def design(self) -> Any:
        return self.packet_plans[0].fd_request.design

    @property
    def prior_snapshot_hash(self) -> str | None:
        if self.prior_fd_snapshot is not self._validated_prior_fd_snapshot:
            raise ValueError("sampled look transaction request was tampered")
        if self.prior_fd_snapshot is None:
            if (
                self._validated_prior_fd_snapshot_hash is not None
                or self._validated_prior_fd_snapshot_custody is not None
            ):
                raise ValueError("sampled look transaction request was tampered")
            return None
        expected_hash = self._validated_prior_fd_snapshot_hash
        custody = self._validated_prior_fd_snapshot_custody
        if (
            expected_hash is None
            or custody is None
            or not _retains_fd_snapshot_custody(
                self.prior_fd_snapshot,
                custody,
                expected_hash,
            )
        ):
            raise ValueError("sampled look transaction request was tampered")
        return expected_hash

    @property
    def prior_corrected_attribution_state_hash(self) -> str | None:
        if (
            self.prior_corrected_attribution_state
            is not self._validated_prior_corrected_attribution_state
        ):
            raise ValueError("sampled look transaction request was tampered")
        if self.prior_corrected_attribution_state is None:
            if (
                self._validated_prior_corrected_attribution_state_hash is not None
                or self._validated_prior_corrected_attribution_custody is not None
            ):
                raise ValueError("sampled look transaction request was tampered")
            return None
        expected_hash = self._validated_prior_corrected_attribution_state_hash
        custody = self._validated_prior_corrected_attribution_custody
        if (
            expected_hash is None
            or custody is None
            or not _retains_moments_custody(
                self.prior_corrected_attribution_state,
                custody,
                expected_hash,
            )
        ):
            raise ValueError("sampled look transaction request was tampered")
        return expected_hash

    @property
    def prior_device_resident_result_hash(self) -> str | None:
        prior = self.prior_device_resident_result
        if prior is not self._validated_prior_device_resident_result:
            raise ValueError("sampled look transaction request was tampered")
        if prior is None:
            if self._validated_prior_device_resident_result_hash is not None:
                raise ValueError("sampled look transaction request was tampered")
            return None
        expected = self._validated_prior_device_resident_result_hash
        if expected is None or prior.result_hash != expected:
            raise ValueError("sampled look transaction request was tampered")
        return expected

    def _current_hash_custody(self, request_hash: str) -> tuple[object, ...]:
        """Capture the shallow frozen-object boundary behind a validated hash.

        The prior moment and FD objects perform their full array integrity scans
        once during construction.  Repeated hash access may trust only these
        exact frozen instances; replacing any transaction field invalidates the
        memoized identity without rescanning the route axis.
        """

        return (
            self.packet_plans,
            tuple(plan.plan_hash for plan in self.packet_plans),
            self.sketch_rank,
            self.registered_sequential_design_hash,
            self.registered_look_index,
            self.registered_look_cumulative_stratum_counts,
            self.prior_corrected_attribution_state,
            self.prior_fd_snapshot,
            self.prior_device_resident_result,
            self.dual_chart_row_group_plan,
            (
                None
                if self.dual_chart_row_group_plan is None
                else self.dual_chart_row_group_plan.group_plan_hash
            ),
            self.transaction_kind,
            self.schema,
            request_hash,
        )

    def _retains_hash_custody(self) -> bool:
        custody = self._request_hash_custody
        if custody is None or len(custody) != 14:
            return False
        retains_public_fields = (
            self.packet_plans is custody[0]
            and self.sketch_rank == custody[2]
            and self.registered_sequential_design_hash == custody[3]
            and self.registered_look_index == custody[4]
            and self.registered_look_cumulative_stratum_counts is custody[5]
            and self.prior_corrected_attribution_state is custody[6]
            and self.prior_fd_snapshot is custody[7]
            and self.prior_device_resident_result is custody[8]
            and self.dual_chart_row_group_plan is custody[9]
            and self.transaction_kind == custody[11]
            and self.schema == custody[12]
            and self._request_hash == custody[13]
        )
        if not retains_public_fields:
            return False
        if tuple(plan.plan_hash for plan in self.packet_plans) != custody[1]:
            return False
        if (
            None
            if self.dual_chart_row_group_plan is None
            else self.dual_chart_row_group_plan.group_plan_hash
        ) != custody[10]:
            return False
        self.prior_corrected_attribution_state_hash
        self.prior_snapshot_hash
        self.prior_device_resident_result_hash
        return True

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "transaction_kind": self.transaction_kind,
            "packet_plan_hashes": tuple(plan.plan_hash for plan in self.packet_plans),
            "sketch_rank": self.sketch_rank,
            "registered_sequential_design_hash": (
                self.registered_sequential_design_hash
            ),
            "registered_look_index": self.registered_look_index,
            "registered_look_cumulative_stratum_counts": (
                self.registered_look_cumulative_stratum_counts
            ),
            "prior_corrected_attribution_state_hash": (
                self.prior_corrected_attribution_state_hash
            ),
            "prior_fd_snapshot_hash": self.prior_snapshot_hash,
            "prior_device_resident_result_hash": (
                self.prior_device_resident_result_hash
            ),
        }
        if self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5:
            assert self.dual_chart_row_group_plan is not None
            payload["dual_chart_row_group_plan_hash"] = (
                self.dual_chart_row_group_plan.group_plan_hash
            )
        elif self.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6:
            assert self.dual_chart_row_group_plan is not None
            payload["dual_chart_row_plan_scientific_hash"] = (
                _device_resident_row_plan_scientific_hash(
                    self.dual_chart_row_group_plan
                )
            )
        return payload

    @property
    def request_hash(self) -> str:
        if not self._retains_hash_custody() or self._request_hash is None:
            raise ValueError("sampled look transaction request was tampered")
        return self._request_hash

    def _constructor_payload(self) -> dict[str, object]:
        return {
            "packet_plans": self.packet_plans,
            "sketch_rank": self.sketch_rank,
            "registered_sequential_design_hash": (
                self.registered_sequential_design_hash
            ),
            "registered_look_index": self.registered_look_index,
            "registered_look_cumulative_stratum_counts": (
                self.registered_look_cumulative_stratum_counts
            ),
            "prior_corrected_attribution_state": (
                self.prior_corrected_attribution_state
            ),
            "prior_fd_snapshot": self.prior_fd_snapshot,
            "prior_device_resident_result": self.prior_device_resident_result,
            "dual_chart_row_group_plan": self.dual_chart_row_group_plan,
            "transaction_kind": self.transaction_kind,
            "schema": self.schema,
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[str, object], str]]:
        return _rebuild_sampled_registered_look_request, (
            self._constructor_payload(),
            self.request_hash,
        )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRegisteredLookReceipt:
    """Typed proof that one look used distinct rows from each capture/VJP.

    This is the only sampled-D object that authorizes the FD snapshot as
    literal-Jacobian input to canonical Q.  It remains discovery evidence and
    cannot certify F/Q/R, open calibration/oracle execution, or commit a mask.
    """

    transaction_request_hash: str
    packet_plan_hashes: tuple[str, ...]
    provider_receipt_hashes: tuple[str, ...]
    corrected_evidence_child_receipt_hashes: tuple[str, ...]
    literal_jacobian_child_receipt_hashes: tuple[str, ...]
    raw_dual_stream_receipt_hashes: tuple[str, ...]
    row_object_separation_receipt_hashes: tuple[str, ...]
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    packet_count: int
    capture_batch_count: int
    vjp_count: int
    same_capture_and_vjp: bool = True
    corrected_rows_excluded_from_response_geometry: bool = True
    proposal_discovery_input_authority: bool = True
    canonical_q_input_authority: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_DUAL_CHART_REGISTERED_LOOK_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        request_hash = _identity(
            self.transaction_request_hash,
            name="dual-chart transaction request hash",
        )
        ledgers = tuple(
            tuple(_identity(value, name=name) for value in values)
            for name, values in (
                ("dual-chart packet-plan hash", self.packet_plan_hashes),
                ("dual-chart provider receipt hash", self.provider_receipt_hashes),
                (
                    "dual-chart corrected-evidence child receipt hash",
                    self.corrected_evidence_child_receipt_hashes,
                ),
                (
                    "dual-chart literal-Jacobian child receipt hash",
                    self.literal_jacobian_child_receipt_hashes,
                ),
                (
                    "dual-chart raw stream receipt hash",
                    self.raw_dual_stream_receipt_hashes,
                ),
                (
                    "dual-chart row-object separation receipt hash",
                    self.row_object_separation_receipt_hashes,
                ),
            )
        )
        packet_count = _positive_int(self.packet_count, name="dual-chart packet count")
        capture_count = _positive_int(
            self.capture_batch_count,
            name="dual-chart capture-batch count",
        )
        vjp_count = _positive_int(self.vjp_count, name="dual-chart VJP count")
        if any(len(values) != packet_count for values in ledgers):
            raise ValueError("dual-chart registered-look receipt is partial")
        if capture_count != packet_count or vjp_count != packet_count:
            raise ValueError(
                "dual-chart registered look requires exactly one capture and VJP "
                "per physical packet"
            )
        evidence = _identity(
            self.corrected_evidence_stream_identity_hash,
            name="corrected-evidence stream identity hash",
        )
        jacobian = _identity(
            self.literal_jacobian_stream_identity_hash,
            name="literal-Jacobian stream identity hash",
        )
        if evidence == jacobian:
            raise ValueError("dual-chart stream identities collapsed")
        if (
            self.same_capture_and_vjp is not True
            or self.corrected_rows_excluded_from_response_geometry is not True
            or self.proposal_discovery_input_authority is not True
            or self.canonical_q_input_authority is not True
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.schema != SAMPLED_DUAL_CHART_REGISTERED_LOOK_RECEIPT_SCHEMA
        ):
            raise ValueError("dual-chart registered-look receipt exceeded its authority")
        object.__setattr__(self, "transaction_request_hash", request_hash)
        object.__setattr__(self, "packet_plan_hashes", ledgers[0])
        object.__setattr__(self, "provider_receipt_hashes", ledgers[1])
        object.__setattr__(
            self,
            "corrected_evidence_child_receipt_hashes",
            ledgers[2],
        )
        object.__setattr__(
            self,
            "literal_jacobian_child_receipt_hashes",
            ledgers[3],
        )
        object.__setattr__(self, "raw_dual_stream_receipt_hashes", ledgers[4])
        object.__setattr__(
            self,
            "row_object_separation_receipt_hashes",
            ledgers[5],
        )
        object.__setattr__(self, "corrected_evidence_stream_identity_hash", evidence)
        object.__setattr__(self, "literal_jacobian_stream_identity_hash", jacobian)
        object.__setattr__(self, "packet_count", packet_count)
        object.__setattr__(self, "capture_batch_count", capture_count)
        object.__setattr__(self, "vjp_count", vjp_count)

    def _hash_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "transaction_request_hash",
                "packet_plan_hashes",
                "provider_receipt_hashes",
                "corrected_evidence_child_receipt_hashes",
                "literal_jacobian_child_receipt_hashes",
                "raw_dual_stream_receipt_hashes",
                "row_object_separation_receipt_hashes",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "packet_count",
                "capture_batch_count",
                "vjp_count",
                "same_capture_and_vjp",
                "corrected_rows_excluded_from_response_geometry",
                "proposal_discovery_input_authority",
                "canonical_q_input_authority",
                "fq_certification_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
            )
        }

    @property
    def receipt_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_DUAL_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("dual-chart registered-look receipt was tampered")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledDualChartRowCustodyRegisteredLookReceipt:
    """Path-free V5 binding from registered reducers to raw A/J custody.

    This receipt proves only that estimand-free corrected-A and literal-J rows
    were staged from the same preregistered packet group and published after
    every provider/reducer proof closed.  It grants no certification, oracle,
    mask, terminal-validation, or Pareto authority.
    """

    transaction_request_hash: str
    registered_packet_plan_hashes: tuple[str, ...]
    dual_chart_receipt_hash: str
    group_plan_hash: str
    group_commit_receipt: V1SampledDualChartRowGroupCommitReceipt
    provider_receipt_hashes: tuple[str, ...]
    corrected_reducer_closure_hashes: tuple[str, ...]
    literal_j_reducer_closure_hashes: tuple[str, ...]
    corrected_custody_branch_receipt_hashes: tuple[str, ...]
    literal_j_custody_branch_receipt_hashes: tuple[str, ...]
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    packet_count: int
    physical_scratch_path_bound: bool = False
    raw_shards_carry_normalized_weights: bool = False
    proposal_input_custody_only: bool = True
    fq_certification_authority: bool = False
    c_stream_authority: bool = False
    k_calibration_authority: bool = False
    o_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_DUAL_CHART_ROW_CUSTODY_REGISTERED_LOOK_RECEIPT_SCHEMA_V5
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.group_commit_receipt) is not V1SampledDualChartRowGroupCommitReceipt:
            raise TypeError("v5 custody receipt requires its exact group receipt")
        self.group_commit_receipt.assert_integrity()
        packet_count = _positive_int(
            self.packet_count,
            name="v5 custody packet count",
        )
        ledgers = tuple(
            tuple(_identity(value, name=name) for value in values)
            for name, values in (
                (
                    "registered packet-plan hash",
                    self.registered_packet_plan_hashes,
                ),
                ("provider receipt hash", self.provider_receipt_hashes),
                (
                    "corrected reducer closure hash",
                    self.corrected_reducer_closure_hashes,
                ),
                (
                    "literal-J reducer closure hash",
                    self.literal_j_reducer_closure_hashes,
                ),
                (
                    "corrected-A custody branch receipt hash",
                    self.corrected_custody_branch_receipt_hashes,
                ),
                (
                    "literal-J custody branch receipt hash",
                    self.literal_j_custody_branch_receipt_hashes,
                ),
            )
        )
        if any(len(values) != packet_count for values in ledgers):
            raise ValueError("v5 custody receipt packet ledgers are partial")
        if any(len(set(values)) != len(values) for values in ledgers):
            raise ValueError("v5 custody receipt ledgers reuse a proof identity")
        evidence = _identity(
            self.corrected_evidence_stream_identity_hash,
            name="v5 corrected-evidence stream identity hash",
        )
        jacobian = _identity(
            self.literal_jacobian_stream_identity_hash,
            name="v5 literal-Jacobian stream identity hash",
        )
        if evidence == jacobian:
            raise ValueError("v5 custody receipt collapsed corrected-A and literal-J")
        for name in (
            "transaction_request_hash",
            "dual_chart_receipt_hash",
            "group_plan_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        expected_reducer_proofs = tuple(
            (a_hash, j_hash)
            for a_hash, j_hash in zip(ledgers[2], ledgers[3], strict=True)
        )
        group = self.group_commit_receipt
        if (
            group.group_plan_hash != self.group_plan_hash
            or group.provider_proof_hashes != ledgers[1]
            or group.reducer_proof_hashes != expected_reducer_proofs
            or group.packet_count != packet_count
        ):
            raise ValueError(
                "v5 custody receipt does not bind the exact group proof closure"
            )
        if (
            self.physical_scratch_path_bound is not False
            or self.raw_shards_carry_normalized_weights is not False
            or self.proposal_input_custody_only is not True
            or any(
                (
                    self.fq_certification_authority,
                    self.c_stream_authority,
                    self.k_calibration_authority,
                    self.o_oracle_authority,
                    self.mask_commit_authority,
                    self.terminal_validation_authority,
                    self.pareto_authority,
                )
            )
            or self.schema
            != SAMPLED_DUAL_CHART_ROW_CUSTODY_REGISTERED_LOOK_RECEIPT_SCHEMA_V5
        ):
            raise ValueError("v5 row-custody receipt exceeded inert custody authority")
        object.__setattr__(self, "registered_packet_plan_hashes", ledgers[0])
        object.__setattr__(self, "provider_receipt_hashes", ledgers[1])
        object.__setattr__(self, "corrected_reducer_closure_hashes", ledgers[2])
        object.__setattr__(self, "literal_j_reducer_closure_hashes", ledgers[3])
        object.__setattr__(
            self,
            "corrected_custody_branch_receipt_hashes",
            ledgers[4],
        )
        object.__setattr__(
            self,
            "literal_j_custody_branch_receipt_hashes",
            ledgers[5],
        )
        object.__setattr__(self, "corrected_evidence_stream_identity_hash", evidence)
        object.__setattr__(self, "literal_jacobian_stream_identity_hash", jacobian)
        object.__setattr__(self, "packet_count", packet_count)
        self.receipt_hash

    def _hash_payload(self) -> dict[str, object]:
        return {
            name: (
                self.group_commit_receipt.receipt_hash
                if name == "group_commit_receipt_hash"
                else getattr(self, name)
            )
            for name in (
                "schema",
                "transaction_request_hash",
                "registered_packet_plan_hashes",
                "dual_chart_receipt_hash",
                "group_plan_hash",
                "group_commit_receipt_hash",
                "provider_receipt_hashes",
                "corrected_reducer_closure_hashes",
                "literal_j_reducer_closure_hashes",
                "corrected_custody_branch_receipt_hashes",
                "literal_j_custody_branch_receipt_hashes",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "packet_count",
                "physical_scratch_path_bound",
                "raw_shards_carry_normalized_weights",
                "proposal_input_custody_only",
                "fq_certification_authority",
                "c_stream_authority",
                "k_calibration_authority",
                "o_oracle_authority",
                "mask_commit_authority",
                "terminal_validation_authority",
                "pareto_authority",
            )
        }

    @property
    def receipt_hash(self) -> str:
        observed = content_hash(
            self._hash_payload(),
            prefix=_DUAL_ROW_CUSTODY_RECEIPT_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("v5 row-custody receipt was tampered")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledRegisteredLookCommit:
    """Immutable all-or-nothing discovery output for one registered look."""

    request: V1SampledRegisteredLookTransactionRequest
    reduced_packets: tuple[V1GPUReducedProposalPacket, ...]
    reduced_closures: tuple[V1SampledProposalDiscoveryClosure, ...]
    fd_append_results: tuple[V1SampledFDAppendResult, ...]
    fd_closures: tuple[V1SampledProposalDiscoveryClosure, ...]
    provider_receipt_hashes: tuple[str, ...]
    corrected_attribution_moments: V1StratifiedCorrectedAttributionMoments
    fd_snapshot: V1StratifiedFDAccumulatorSnapshot
    reduced_packet_hashes: tuple[str, ...] | None = None
    dual_chart_receipt: V1SampledDualChartRegisteredLookReceipt | None = None
    dual_chart_row_custody_receipt: (
        V1SampledDualChartRowCustodyRegisteredLookReceipt | None
    ) = None
    authority_scope: str = SAMPLED_DISCOVERY_AUTHORITY_SCOPE
    proposal_discovery_input_authority: bool = True
    canonical_q_input_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA
    _trust_cached_reduced_packet_hashes: bool = field(
        default=False,
        repr=False,
        compare=False,
    )
    _commit_hash: str | None = field(init=False, default=None, repr=False)
    _commit_hash_custody: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.request) is not V1SampledRegisteredLookTransactionRequest:
            raise TypeError("sampled look commit requires its exact request")
        self.request.request_hash
        count = len(self.request.packet_plans)
        packets = tuple(self.reduced_packets)
        reduced = tuple(self.reduced_closures)
        append_results = tuple(self.fd_append_results)
        fd = tuple(self.fd_closures)
        receipts = tuple(
            _identity(value, name="sampled look provider receipt hash")
            for value in self.provider_receipt_hashes
        )
        expected_schema = _commit_schema_for_request_schema(self.request.schema)
        if self.schema != expected_schema:
            raise ValueError("sampled look commit schema is invalid")
        bounded_memory = (
            self.request.schema
            in (
                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3,
                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
            )
        )
        dual_chart = self.request.schema in (
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
        )
        row_custody = (
            self.request.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5
        )
        supplied_packet_hashes = (
            None
            if self.reduced_packet_hashes is None
            else tuple(
                _identity(value, name="sampled reduced packet hash")
                for value in self.reduced_packet_hashes
            )
        )
        if not (
            len(reduced)
            == len(append_results)
            == len(fd)
            == len(receipts)
            == count
        ):
            raise ValueError("sampled look commit is partial")
        if bounded_memory:
            if packets:
                raise ValueError(
                    "bounded-memory sampled look commit retained reduced packets"
                )
        elif len(packets) != count:
            raise ValueError(
                "materialized sampled look commit packet ledger is partial"
            )
        if supplied_packet_hashes is not None and len(supplied_packet_hashes) != count:
            raise ValueError("sampled look commit packet-hash ledger is partial")
        if bounded_memory and supplied_packet_hashes is None:
            raise ValueError(
                "bounded-memory sampled look commit lacks its packet-hash ledger"
            )
        observed_packet_hashes: list[str] = []
        for index, (
            plan,
            score_closure,
            append_result,
            fd_closure,
        ) in enumerate(
            zip(self.request.packet_plans, reduced, append_results, fd)
        ):
            if packets:
                packet = packets[index]
                if type(packet) is not V1GPUReducedProposalPacket:
                    raise TypeError("sampled look commit requires exact reduced packets")
                observed_hash = packet.assert_integrity(
                    trust_cached_immutable=self._trust_cached_reduced_packet_hashes,
                )
                if (
                    packet.request.request_hash
                    != plan.reduced_request.packet_request.request_hash
                ):
                    raise ValueError("sampled look packet differs from its frozen plan")
                if packet.request.packet_index != (
                    self.request.packet_plans[0].fd_request.packet_index + index
                ):
                    raise ValueError(
                        "sampled look committed packets are out of order"
                    )
            else:
                assert supplied_packet_hashes is not None
                observed_hash = supplied_packet_hashes[index]
            observed_packet_hashes.append(observed_hash)
            if type(score_closure) is not V1SampledProposalDiscoveryClosure or (
                score_closure.adapter_kind != "gpu_reduced_corrected_attribution_packet"
            ):
                raise TypeError("sampled look reduced closure is invalid")
            if type(fd_closure) is not V1SampledProposalDiscoveryClosure or (
                fd_closure.adapter_kind != "same_stratum_fd_packet_append"
            ):
                raise TypeError("sampled look FD closure is invalid")
            if type(append_result) is not V1SampledFDAppendResult:
                raise TypeError("sampled look commit requires exact FD append results")
            if (
                append_result.request_hash != plan.fd_request.request_hash
                or append_result.design_hash != plan.fd_request.design.design_hash
                or append_result.packet_boundary_index_before
                != plan.fd_request.packet_index
                or append_result.packet_boundary_index_after
                != plan.fd_request.packet_index + 1
            ):
                raise ValueError("sampled look FD append result differs from its plan")
            if score_closure.child_result_hash != observed_hash:
                raise ValueError(
                    "sampled look reduced closure does not bind its packet"
                )
            if score_closure.adapter_request_hash != plan.reduced_request.request_hash:
                raise ValueError("sampled look reduced closure does not bind its plan")
            if fd_closure.adapter_request_hash != plan.fd_request.request_hash:
                raise ValueError("sampled look FD closure does not bind its plan")
            if fd_closure.child_result_hash != append_result.result_hash:
                raise ValueError("sampled look FD closure does not bind its append")
            if dual_chart and (
                score_closure.raw_logit_stream_identity_hash
                != plan.corrected_evidence_stream_identity_hash
                or fd_closure.raw_logit_stream_identity_hash
                != plan.literal_jacobian_stream_identity_hash
            ):
                raise ValueError(
                    "sampled dual-chart closures swapped or collapsed their streams"
                )
            score_closure.closure_hash
            fd_closure.closure_hash
        packet_hashes = tuple(observed_packet_hashes)
        if supplied_packet_hashes is not None and supplied_packet_hashes != packet_hashes:
            raise ValueError("sampled look commit packet hashes disagree")
        if type(self.fd_snapshot) is not V1StratifiedFDAccumulatorSnapshot:
            raise TypeError("sampled look commit requires an exact FD snapshot")
        self.fd_snapshot.assert_integrity()
        if (
            type(self.corrected_attribution_moments)
            is not V1StratifiedCorrectedAttributionMoments
        ):
            raise TypeError("sampled look commit requires exact corrected moments")
        self.corrected_attribution_moments.assert_integrity()
        if (
            self.fd_snapshot.design_hash != self.request.design.design_hash
            or self.fd_snapshot.sketch_rank != self.request.sketch_rank
            or self.fd_snapshot.packet_boundary_index
            != self.request.packet_plans[0].fd_request.packet_index + count
        ):
            raise ValueError("sampled look FD boundary does not close every packet")
        if (
            tuple(self.corrected_attribution_moments.stratum_counts)
            != self.request.registered_look_cumulative_stratum_counts
            or self.corrected_attribution_moments.look_index
            != self.request.registered_look_index
            or self.corrected_attribution_moments.prior_state_hash
            != self.request.prior_corrected_attribution_state_hash
            or self.corrected_attribution_moments.route_axis_hash
            != self.request.packet_plans[
                0
            ].reduced_request.packet_request.route_axis_hash
            or self.corrected_attribution_moments.current_mask_hash
            != self.request.design.current_mask_hash
        ):
            raise ValueError(
                "sampled look corrected moments do not close the registered look"
            )
        snapshot_context_ledger = tuple(
            (name, tuple(child.context_ids))
            for name, child in self.fd_snapshot.stratum_snapshots
        )
        snapshot_source_ledger = self.fd_snapshot.source_measurement_hashes_by_stratum
        final_append = append_results[-1]
        if final_append.context_ledger_hash_after != content_hash(
            snapshot_context_ledger, prefix=_FD_CHILD_LEDGER_PREFIX
        ) or final_append.source_measurement_ledger_hash_after != content_hash(
            snapshot_source_ledger, prefix=_FD_SOURCE_AXIS_PREFIX
        ):
            raise ValueError("sampled look final FD append does not bind its snapshot")
        snapshot_counts = {
            name: child.context_count
            for name, child in self.fd_snapshot.stratum_snapshots
        }
        running_counts = {
            name: 0 for name in self.request.design.stratum_names
        }
        if self.request.prior_fd_snapshot is not None:
            running_counts.update(
                {
                    name: child.context_count
                    for name, child in (
                        self.request.prior_fd_snapshot.stratum_snapshots
                    )
                }
            )
        for plan, append_result in zip(self.request.packet_plans, append_results):
            stratum = plan.fd_request.stratum_labels[0]
            running_counts[stratum] += len(plan.fd_request.context_ids)
            if (
                append_result.target_stratum_context_count_after
                != running_counts[stratum]
            ):
                raise ValueError(
                    "sampled look FD stratum count differs from its packet boundary"
                )
        if running_counts != snapshot_counts:
            raise ValueError("sampled look FD packet chain does not close its snapshot")
        if dual_chart:
            dual_receipt = self.dual_chart_receipt
            if type(dual_receipt) is not V1SampledDualChartRegisteredLookReceipt:
                raise TypeError(
                    "canonical sampled commit requires its exact dual-chart receipt"
                )
            if (
                dual_receipt.transaction_request_hash != self.request.request_hash
                or dual_receipt.packet_plan_hashes
                != tuple(plan.plan_hash for plan in self.request.packet_plans)
                or dual_receipt.provider_receipt_hashes != receipts
                or dual_receipt.corrected_evidence_child_receipt_hashes
                != tuple(value.adapter_receipt_hash for value in reduced)
                or dual_receipt.literal_jacobian_child_receipt_hashes
                != tuple(value.adapter_receipt_hash for value in fd)
                or dual_receipt.corrected_evidence_stream_identity_hash
                != self.request.packet_plans[
                    0
                ].corrected_evidence_stream_identity_hash
                or dual_receipt.literal_jacobian_stream_identity_hash
                != self.request.packet_plans[0].literal_jacobian_stream_identity_hash
            ):
                raise ValueError(
                    "sampled commit dual-chart receipt does not bind its children"
                )
            dual_receipt.receipt_hash
            if self.canonical_q_input_authority is not True:
                raise ValueError(
                    "canonical sampled commit lost literal-Jacobian Q-input authority"
                )
        elif (
            self.dual_chart_receipt is not None
            or self.canonical_q_input_authority is not False
        ):
            raise ValueError(
                "legacy sampled commits cannot claim canonical-Q input authority"
            )
        if row_custody:
            custody_receipt = self.dual_chart_row_custody_receipt
            if type(custody_receipt) is not (
                V1SampledDualChartRowCustodyRegisteredLookReceipt
            ):
                raise TypeError(
                    "sampled v5 commit requires its exact path-free custody receipt"
                )
            assert self.dual_chart_receipt is not None
            if (
                custody_receipt.transaction_request_hash
                != self.request.request_hash
                or custody_receipt.registered_packet_plan_hashes
                != tuple(plan.plan_hash for plan in self.request.packet_plans)
                or custody_receipt.dual_chart_receipt_hash
                != self.dual_chart_receipt.receipt_hash
                or custody_receipt.group_plan_hash
                != self.request.dual_chart_row_group_plan.group_plan_hash
                or custody_receipt.provider_receipt_hashes != receipts
                or custody_receipt.corrected_reducer_closure_hashes
                != tuple(value.closure_hash for value in reduced)
                or custody_receipt.literal_j_reducer_closure_hashes
                != tuple(value.closure_hash for value in fd)
            ):
                raise ValueError(
                    "sampled v5 commit custody receipt does not bind its exact "
                    "registered children"
                )
            custody_receipt.receipt_hash
        elif self.dual_chart_row_custody_receipt is not None:
            raise ValueError("sampled v1-v4 commits cannot claim v5 row custody")
        if (
            self.authority_scope != SAMPLED_DISCOVERY_AUTHORITY_SCOPE
            or self.proposal_discovery_input_authority is not True
            or any(
                (
                    self.fq_certification_authority,
                    self.bce_oracle_authority,
                    self.mask_commit_authority,
                )
            )
        ):
            raise ValueError("sampled look commit exceeded discovery authority")
        object.__setattr__(self, "reduced_packets", packets)
        object.__setattr__(self, "reduced_packet_hashes", packet_hashes)
        object.__setattr__(self, "reduced_closures", reduced)
        object.__setattr__(self, "fd_append_results", append_results)
        object.__setattr__(self, "fd_closures", fd)
        object.__setattr__(self, "provider_receipt_hashes", receipts)
        observed_commit_hash = content_hash(
            self._hash_payload(),
            prefix=_COMMIT_PREFIX,
        )
        object.__setattr__(self, "_commit_hash", observed_commit_hash)
        object.__setattr__(
            self,
            "_commit_hash_custody",
            self._current_hash_custody(observed_commit_hash),
        )

    def _current_hash_custody(self, commit_hash: str) -> tuple[object, ...]:
        return (
            self.request,
            self.reduced_packets,
            self.reduced_packet_hashes,
            self.reduced_closures,
            tuple(closure.closure_hash for closure in self.reduced_closures),
            self.fd_append_results,
            tuple(result.result_hash for result in self.fd_append_results),
            self.fd_closures,
            tuple(closure.closure_hash for closure in self.fd_closures),
            self.provider_receipt_hashes,
            self.corrected_attribution_moments,
            self.corrected_attribution_moments.state_hash,
            self.fd_snapshot,
            self.fd_snapshot.snapshot_hash,
            self.dual_chart_receipt,
            (
                None
                if self.dual_chart_receipt is None
                else self.dual_chart_receipt.receipt_hash
            ),
            self.dual_chart_row_custody_receipt,
            (
                None
                if self.dual_chart_row_custody_receipt is None
                else self.dual_chart_row_custody_receipt.receipt_hash
            ),
            self.authority_scope,
            self.proposal_discovery_input_authority,
            self.canonical_q_input_authority,
            self.fq_certification_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.schema,
            self._trust_cached_reduced_packet_hashes,
            commit_hash,
        )

    def _retains_hash_custody(self) -> bool:
        custody = self._commit_hash_custody
        if custody is None or len(custody) != 27:
            return False
        retains_public_fields = (
            self.request is custody[0]
            and self.reduced_packets is custody[1]
            and self.reduced_closures is custody[3]
            and self.fd_append_results is custody[5]
            and self.fd_closures is custody[7]
            and self.provider_receipt_hashes is custody[9]
            and self.corrected_attribution_moments is custody[10]
            and self.fd_snapshot is custody[12]
            and self.dual_chart_receipt is custody[14]
            and self.dual_chart_row_custody_receipt is custody[16]
            and self.authority_scope == custody[18]
            and self.proposal_discovery_input_authority is custody[19]
            and self.canonical_q_input_authority is custody[20]
            and self.fq_certification_authority is custody[21]
            and self.bce_oracle_authority is custody[22]
            and self.mask_commit_authority is custody[23]
            and self.schema == custody[24]
            and self._trust_cached_reduced_packet_hashes is custody[25]
            and self._commit_hash == custody[26]
        )
        if not retains_public_fields:
            return False
        return (
            self.reduced_packet_hashes == custody[2]
            and tuple(
                closure.closure_hash for closure in self.reduced_closures
            )
            == custody[4]
            and tuple(result.result_hash for result in self.fd_append_results)
            == custody[6]
            and tuple(closure.closure_hash for closure in self.fd_closures)
            == custody[8]
            and self.corrected_attribution_moments.state_hash == custody[11]
            and self.fd_snapshot.snapshot_hash == custody[13]
            and (
                None
                if self.dual_chart_receipt is None
                else self.dual_chart_receipt.receipt_hash
            )
            == custody[15]
            and (
                None
                if self.dual_chart_row_custody_receipt is None
                else self.dual_chart_row_custody_receipt.receipt_hash
            )
            == custody[17]
        )

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "request_hash": self.request.request_hash,
            "reduced_packet_hashes": self.reduced_packet_hashes,
            "reduced_closure_hashes": tuple(
                closure.closure_hash for closure in self.reduced_closures
            ),
            "fd_append_result_hashes": tuple(
                result.result_hash for result in self.fd_append_results
            ),
            "fd_closure_hashes": tuple(
                closure.closure_hash for closure in self.fd_closures
            ),
            "provider_receipt_hashes": self.provider_receipt_hashes,
            "corrected_attribution_moments_hash": (
                self.corrected_attribution_moments.state_hash
            ),
            "moments_transition_hash": content_hash(
                {
                    "prior_state_hash": (
                        self.request.prior_corrected_attribution_state_hash
                    ),
                    "packet_hashes": self.reduced_packet_hashes,
                    "result_state_hash": (
                        self.corrected_attribution_moments.state_hash
                    ),
                    "arithmetic": (
                        "registered_packet_order_chan_merge_v1"
                        if self.request.schema
                        == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V1
                        else (
                            (
                                (
                                    "bounded_dual_chart_row_custody_registered_"
                                    "transport_chunk_order_chan_merge_v5"
                                    if self.request.schema
                                    == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5
                                    else "bounded_dual_chart_registered_transport_"
                                    "chunk_order_chan_merge_v4"
                                )
                                if self.request.schema
                                in (
                                    SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
                                    SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
                                )
                                else "bounded_incremental_registered_transport_"
                                "chunk_order_chan_merge_v3"
                            )
                            if self.request.schema
                            in (
                                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3,
                                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
                                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
                            )
                            else "registered_transport_chunk_order_chan_merge_v2"
                        )
                    ),
                },
                prefix=_COMMIT_PREFIX + "moments-transition-",
            ),
            "fd_snapshot_hash": self.fd_snapshot.snapshot_hash,
            "authority_scope": self.authority_scope,
            "proposal_discovery_input_authority": (
                self.proposal_discovery_input_authority
            ),
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }
        if self.request.schema in (
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
        ):
            payload.update(
                {
                    "dual_chart_receipt_hash": self.dual_chart_receipt.receipt_hash,
                    "canonical_q_input_authority": (
                        self.canonical_q_input_authority
                    ),
                }
            )
        if self.request.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5:
            assert self.dual_chart_row_custody_receipt is not None
            payload["dual_chart_row_custody_receipt_hash"] = (
                self.dual_chart_row_custody_receipt.receipt_hash
            )
        return payload

    @property
    def commit_hash(self) -> str:
        if not self._retains_hash_custody() or self._commit_hash is None:
            raise ValueError("sampled look commit was tampered")
        self.request.request_hash
        return self._commit_hash

    def _constructor_payload(self) -> dict[str, object]:
        return {
            "request": self.request,
            "reduced_packets": self.reduced_packets,
            "reduced_packet_hashes": self.reduced_packet_hashes,
            "reduced_closures": self.reduced_closures,
            "fd_append_results": self.fd_append_results,
            "fd_closures": self.fd_closures,
            "provider_receipt_hashes": self.provider_receipt_hashes,
            "corrected_attribution_moments": self.corrected_attribution_moments,
            "fd_snapshot": self.fd_snapshot,
            "dual_chart_receipt": self.dual_chart_receipt,
            "dual_chart_row_custody_receipt": (
                self.dual_chart_row_custody_receipt
            ),
            "authority_scope": self.authority_scope,
            "proposal_discovery_input_authority": (
                self.proposal_discovery_input_authority
            ),
            "canonical_q_input_authority": self.canonical_q_input_authority,
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "schema": self.schema,
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[str, object], str]]:
        return _rebuild_sampled_registered_look_commit, (
            self._constructor_payload(),
            self.commit_hash,
        )


def _rebuild_sampled_registered_look_request(
    payload: dict[str, object],
    expected_hash: str,
) -> V1SampledRegisteredLookTransactionRequest:
    request = V1SampledRegisteredLookTransactionRequest(
        **payload,  # type: ignore[arg-type]
    )
    if request.request_hash != expected_hash:
        raise ValueError("sampled look transaction request hash does not match")
    return request


def _rebuild_sampled_registered_look_commit(
    payload: dict[str, object],
    expected_hash: str,
) -> V1SampledRegisteredLookCommit:
    # A deserialized boundary is cold.  It must revalidate every reduced
    # packet rather than inherit the producer's in-process packet trust flag.
    commit = V1SampledRegisteredLookCommit(
        **payload,  # type: ignore[arg-type]
        _trust_cached_reduced_packet_hashes=False,
    )
    if commit.commit_hash != expected_hash:
        raise ValueError("sampled look commit hash does not match")
    return commit


@dataclass(frozen=True, slots=True)
class _VerifiedProviderPacket:
    provider_receipt_hash: str
    corrected_evidence_child_receipt_hash: str | None = None
    literal_jacobian_child_receipt_hash: str | None = None
    raw_dual_stream_receipt_hash: str | None = None
    row_object_separation_receipt_hash: str | None = None
    corrected_custody_branch_receipt_hash: str | None = None
    literal_j_custody_branch_receipt_hash: str | None = None
    capture_batch_count: int = 0
    vjp_count: int = 0


@dataclass(frozen=True, slots=True)
class V1SampledDualChartCustodyProviderVerificationReceipt:
    """Array-free proof that one direct A/J custody scorer call closed.

    This receipt deliberately excludes score/FD reducer authority.  It proves
    only that one preregistered provider request emitted distinct corrected-A
    and literal-J children from one capture/VJP and that both complete row
    streams reached the private row-group transaction.
    """

    group_plan_hash: str
    group_packet_plan_hash: str
    packet_ordinal: int
    provider_plan_identity_hash: str
    source_capture_identity_hash: str
    provider_receipt_hash: str
    raw_dual_stream_receipt_hash: str
    row_object_separation_receipt_hash: str
    corrected_custody_branch_receipt_hash: str
    literal_j_custody_branch_receipt_hash: str
    route_window_count: int
    capture_batch_count: int = 1
    vjp_count: int = 1
    direct_dual_custody_only: bool = True
    proposal_input_row_custody_authority: bool = True
    score_or_fd_reducer_authority: bool = False
    d_stability_authority: bool = False
    fq_certification_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    schema: str = SAMPLED_DUAL_CHART_CUSTODY_PROVIDER_VERIFICATION_SCHEMA_V1
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _CUSTODY_PROVIDER_VERIFICATION_ISSUER_TOKEN:
            raise PermissionError(
                "dual-chart custody provider verifications are verifier-issued only"
            )
        for name in (
            "group_plan_hash",
            "group_packet_plan_hash",
            "provider_plan_identity_hash",
            "source_capture_identity_hash",
            "provider_receipt_hash",
            "raw_dual_stream_receipt_hash",
            "row_object_separation_receipt_hash",
            "corrected_custody_branch_receipt_hash",
            "literal_j_custody_branch_receipt_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        ordinal = int(self.packet_ordinal)
        windows = int(self.route_window_count)
        if (
            isinstance(self.packet_ordinal, bool)
            or ordinal < 0
            or isinstance(self.route_window_count, bool)
            or windows <= 0
            or self.capture_batch_count != 1
            or self.vjp_count != 1
            or self.direct_dual_custody_only is not True
            or self.proposal_input_row_custody_authority is not True
            or any(
                value is not False
                for value in (
                    self.score_or_fd_reducer_authority,
                    self.d_stability_authority,
                    self.fq_certification_authority,
                    self.c_k_o_authority,
                    self.mask_commit_authority,
                    self.terminal_or_pareto_authority,
                )
            )
            or self.schema
            != SAMPLED_DUAL_CHART_CUSTODY_PROVIDER_VERIFICATION_SCHEMA_V1
        ):
            raise ValueError(
                "dual-chart custody provider verification crossed its authority "
                "or one-capture/one-VJP boundary"
            )
        object.__setattr__(self, "packet_ordinal", ordinal)
        object.__setattr__(self, "route_window_count", windows)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_CUSTODY_PROVIDER_VERIFICATION_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError(
                "dual-chart custody provider verification changed after issuance"
            )
        if expected_hash is not None and observed != expected_hash:
            raise ValueError(
                "dual-chart custody provider verification hash differs"
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDTransactionResult:
    """Scalar/hash-only registered-look result for the live sampled-D path."""

    transaction_request_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    registered_look_index: int
    look_receipts: tuple[V1SampledDeviceResidentDLookReceipt, ...]
    provider_receipt_hashes: tuple[str, ...]
    prior_result_hash: str | None = None
    cumulative_packet_count_after: int = 0
    cumulative_stratum_counts_after: tuple[int, ...] = ()
    cumulative_look_receipt_hashes: tuple[str, ...] = ()
    live_session_custody_hash: str | None = None
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    commit_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_DEVICE_RESIDENT_D_TRANSACTION_RESULT_SCHEMA_V1

    def __post_init__(self) -> None:
        for name in (
            "transaction_request_hash",
            "session_design_hash",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "registered_look_index",
            _positive_int(self.registered_look_index, name="registered look index"),
        )
        receipts = tuple(self.look_receipts)
        if not receipts or any(
            type(receipt) is not V1SampledDeviceResidentDLookReceipt
            for receipt in receipts
        ):
            raise TypeError("device-resident D transaction requires exact look receipts")
        for ordinal, receipt in enumerate(receipts):
            receipt.assert_integrity()
            if (
                receipt.session_design_hash != self.session_design_hash
                or receipt.hbm_forecast_hash != self.hbm_forecast_hash
                or receipt.registered_look_index != self.registered_look_index
                or receipt.packet_ordinal != ordinal
                or receipt.route_scale_d2h_bytes != 0
                or receipt.route_scale_host_tensor_materializations != 0
            ):
                raise ValueError(
                    "device-resident D look receipt is not an in-order scalar receipt"
                )
        object.__setattr__(self, "look_receipts", receipts)
        object.__setattr__(
            self,
            "provider_receipt_hashes",
            _identity_axis(
                self.provider_receipt_hashes,
                name="device-resident D provider receipt hashes",
                unique=True,
            ),
        )
        if len(self.provider_receipt_hashes) != len(receipts):
            raise ValueError("device-resident D provider ledger is partial")
        prior_hash = self.prior_result_hash
        if prior_hash is not None:
            prior_hash = _identity(prior_hash, name="prior device-resident result hash")
        object.__setattr__(self, "prior_result_hash", prior_hash)
        packet_count = _positive_int(
            self.cumulative_packet_count_after,
            name="cumulative device-resident packet count",
        )
        if packet_count < len(receipts):
            raise ValueError("device-resident cumulative packet count is partial")
        object.__setattr__(self, "cumulative_packet_count_after", packet_count)
        counts = tuple(
            _nonnegative_int(value, name="cumulative device-resident stratum count")
            for value in self.cumulative_stratum_counts_after
        )
        if counts != receipts[-1].stratum_counts_after:
            raise ValueError(
                "device-resident cumulative counts differ from the live session"
            )
        object.__setattr__(self, "cumulative_stratum_counts_after", counts)
        cumulative_receipts = _identity_axis(
            self.cumulative_look_receipt_hashes,
            name="cumulative device-resident look receipt hashes",
            unique=True,
        )
        current_receipts = tuple(receipt.receipt_hash for receipt in receipts)
        if (
            len(cumulative_receipts) != packet_count
            or cumulative_receipts[-len(current_receipts) :] != current_receipts
        ):
            raise ValueError(
                "device-resident cumulative look ledger does not end in this result"
            )
        object.__setattr__(
            self,
            "cumulative_look_receipt_hashes",
            cumulative_receipts,
        )
        custody_hash = _identity(
            self.live_session_custody_hash,
            name="live device-resident session custody hash",
        )
        expected_custody = content_hash(
            {
                "session_design_hash": self.session_design_hash,
                "hbm_forecast_hash": self.hbm_forecast_hash,
            },
            prefix="partition-free-v1-sampled-device-resident-D-live-custody-",
        )
        if custody_hash != expected_custody:
            raise ValueError("device-resident result names another live-session custody")
        object.__setattr__(self, "live_session_custody_hash", custody_hash)
        object.__setattr__(
            self,
            "route_scale_d2h_bytes",
            int(self.route_scale_d2h_bytes),
        )
        object.__setattr__(
            self,
            "route_scale_host_tensor_materializations",
            int(self.route_scale_host_tensor_materializations),
        )
        flags = (
            self.commit_authority,
            self.fq_certification_authority,
            self.bce_oracle_authority,
            self.pareto_authority,
        )
        if (
            self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
            or any(type(flag) is not bool for flag in flags)
            or any(flags)
            or self.schema != SAMPLED_DEVICE_RESIDENT_D_TRANSACTION_RESULT_SCHEMA_V1
        ):
            raise ValueError(
                "device-resident D transaction result crossed its authority boundary"
            )

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "transaction_request_hash": self.transaction_request_hash,
                "session_design_hash": self.session_design_hash,
                "hbm_forecast_hash": self.hbm_forecast_hash,
                "registered_look_index": self.registered_look_index,
                "look_receipt_hashes": tuple(
                    receipt.receipt_hash for receipt in self.look_receipts
                ),
                "provider_receipt_hashes": self.provider_receipt_hashes,
                "prior_result_hash": self.prior_result_hash,
                "cumulative_packet_count_after": self.cumulative_packet_count_after,
                "cumulative_stratum_counts_after": (
                    self.cumulative_stratum_counts_after
                ),
                "cumulative_look_receipt_hashes": (
                    self.cumulative_look_receipt_hashes
                ),
                "live_session_custody_hash": self.live_session_custody_hash,
                "route_scale_d2h_bytes": self.route_scale_d2h_bytes,
                "route_scale_host_tensor_materializations": (
                    self.route_scale_host_tensor_materializations
                ),
                "all_downstream_authority": False,
            },
            prefix=_DEVICE_RESIDENT_D_TRANSACTION_RESULT_PREFIX,
        )


class _DualChartChunkObjectSeparationWitness:
    """Prove paired child chunks are distinct while both objects are live.

    Bare ``id`` ledgers are not evidence: CPython may recycle an object's id
    after the first child releases it.  This witness retains exactly one
    unpaired tensor, compares the paired tensor with ``is not`` while both are
    simultaneously live, and then releases the pair.  Its state is O(1) in
    route count and its receipt is a deterministic hash chain over windows.
    """

    _ROLES: Final = frozenset(("corrected_evidence", "literal_jacobian"))

    def __init__(self) -> None:
        self._pending_role: str | None = None
        self._pending_window: tuple[int, int] | None = None
        self._pending_rows: object | None = None
        self._pair_count = 0
        self._chain_hash = content_hash(
            {
                "schema": "partition_free_v1_dual_chunk_object_separation_seed_v1",
                "bounded_pending_object_count": 1,
            },
            prefix=_DUAL_RECEIPT_PREFIX + "row-object-separation-seed-",
        )
        self._closed = False

    def observe(self, role: str, start: int, stop: int, rows: Any) -> None:
        if self._closed:
            raise RuntimeError("dual-chunk object witness is already closed")
        if role not in self._ROLES:
            raise ValueError("dual-chunk object witness received another role")
        window = (int(start), int(stop))
        if self._pending_rows is None:
            self._pending_role = role
            self._pending_window = window
            self._pending_rows = rows
            return
        if (
            self._pending_role == role
            or self._pending_window != window
        ):
            raise RuntimeError(
                "dual-chart child chunks were not emitted as exact paired windows"
            )
        if self._pending_rows is rows:
            raise RuntimeError(
                "corrected evidence rows were reused as literal-Jacobian Q rows"
            )
        self._pair_count += 1
        self._chain_hash = content_hash(
            {
                "schema": "partition_free_v1_dual_chunk_object_separation_link_v1",
                "prior_chain_hash": self._chain_hash,
                "pair_index": self._pair_count,
                "route_window": window,
                "roles": tuple(sorted((self._pending_role, role))),
                "simultaneously_live_distinct_objects": True,
            },
            prefix=_DUAL_RECEIPT_PREFIX + "row-object-separation-link-",
        )
        self._pending_role = None
        self._pending_window = None
        self._pending_rows = None

    def close(self, *, expected_pair_count: int) -> str:
        if self._closed:
            raise RuntimeError("dual-chunk object witness is already closed")
        expected = _positive_int(
            expected_pair_count,
            name="dual-chunk expected pair count",
        )
        if self._pending_rows is not None or self._pair_count != expected:
            raise RuntimeError(
                "dual-chart child chunk pairing is partial or out of order"
            )
        self._closed = True
        return content_hash(
            {
                "schema": "partition_free_v1_dual_chunk_object_separation_receipt_v1",
                "pair_count": self._pair_count,
                "chain_hash": self._chain_hash,
                "bounded_pending_object_count": 1,
                "simultaneously_live_distinct_objects": True,
            },
            prefix=_DUAL_RECEIPT_PREFIX + "row-object-separation-receipt-",
        )


class _DeviceResidentDSessionBranchConsumer(_RouteChunkedLossOnlyConsumer):
    """One branch of the live sampled-D same-capture stream bridge."""

    positive_removal_stream_coordinate = RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE

    def __init__(
        self,
        *,
        bridge: _DeviceResidentDSessionPacketBridge,
        role: str,
        stream_identity_hash: str,
    ) -> None:
        if role not in {"corrected_a", "literal_j"}:
            raise ValueError("device-resident D branch role is invalid")
        self._bridge = bridge
        self._role = role
        self.route_ids = bridge.route_ids
        self.device = bridge.device
        self.loss_pullback_hash = _identity(
            stream_identity_hash,
            name="device-resident D branch stream identity hash",
        )
        self.head_ids = (f"{role}:raw_target_logit",)
        self.chunk_size = bridge.registered_route_chunk_size

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        self._bridge.bind_route_chunk_size(self._role, chunk_size)
        self.chunk_size = int(chunk_size)

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        self._bridge.begin_branch(
            self._role,
            context_ids=context_ids,
            route_ids=route_ids,
            destination_transport_hash=destination_transport_hash,
            stream_identity_hash=loss_pullback_hash,
        )

    def consume_positive_removal_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        self._bridge.consume_branch(self._role, start=start, stop=stop, rows=rows)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        raise PermissionError(
            "device-resident D session accepts only raw-logit positive-removal rows"
        )

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        raise PermissionError(
            "device-resident D session refuses consumer-side pullback formation"
        )

    def end_packet(self) -> None:
        self._bridge.end_branch(self._role)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        return self._bridge.branch_receipt(self._role)


class _DeviceResidentDSessionPacketBridge:
    """Pair one packet's corrected-A/literal-J rows into a live D session.

    This is the zero-publication sibling of the V5 row-custody bridge: it
    verifies exact paired windows, consumes each row chunk once into the live
    D owner, and emits only scalar/hash receipts until D closes.
    """

    _ROLES: Final = ("corrected_a", "literal_j")

    def __init__(
        self,
        *,
        transaction_request_hash: str,
        registered_look_index: int,
        packet_ordinal: int,
        plan: V1SampledLookPacketPlan,
        session: V1SampledDeviceResidentDSession,
    ) -> None:
        self.transaction_request_hash = _identity(
            transaction_request_hash,
            name="device-resident D transaction request hash",
        )
        self.registered_look_index = _positive_int(
            registered_look_index,
            name="device-resident D registered look index",
        )
        self.packet_ordinal = int(packet_ordinal)
        if self.packet_ordinal < 0:
            raise ValueError("device-resident D packet ordinal must be nonnegative")
        if type(plan) is not V1SampledLookPacketPlan:
            raise TypeError("device-resident D bridge requires exact packet plan")
        if type(session) is not V1SampledDeviceResidentDSession:
            raise TypeError("device-resident D bridge requires exact live session")
        self.plan = plan
        self.session = session
        packet = plan.reduced_request.packet_request
        self.route_ids = tuple(plan.reduced_request.route_ids)
        self.device = plan.reduced_request.declared_device
        self.route_axis_hash = packet.route_axis_hash
        self.destination_transport_hash = plan.reduced_request.destination_transport_hash
        self.registered_route_chunk_size = packet.registered_route_chunk_size
        self.stream_identities = {
            "corrected_a": plan.corrected_evidence_stream_identity_hash,
            "literal_j": plan.literal_jacobian_stream_identity_hash,
        }
        if (
            not plan.same_capture_and_vjp_required
            or self.stream_identities["corrected_a"]
            == self.stream_identities["literal_j"]
        ):
            raise ValueError("device-resident D requires distinct same-VJP streams")
        self._bound_roles: set[str] = set()
        self._open_roles: set[str] = set()
        self._closed_roles: set[str] = set()
        self._next_route_by_role = {role: 0 for role in self._ROLES}
        self._windows_by_role: dict[str, list[tuple[int, int]]] = {
            role: [] for role in self._ROLES
        }
        self._pending_role: str | None = None
        self._pending_window: tuple[int, int] | None = None
        self._pending_rows: Any | None = None
        self._session_packet_open = False
        self._look_receipt: V1SampledDeviceResidentDLookReceipt | None = None

    @property
    def evidence_consumer(self) -> _DeviceResidentDSessionBranchConsumer:
        return _DeviceResidentDSessionBranchConsumer(
            bridge=self,
            role="corrected_a",
            stream_identity_hash=self.stream_identities["corrected_a"],
        )

    @property
    def jacobian_consumer(self) -> _DeviceResidentDSessionBranchConsumer:
        return _DeviceResidentDSessionBranchConsumer(
            bridge=self,
            role="literal_j",
            stream_identity_hash=self.stream_identities["literal_j"],
        )

    @property
    def look_receipt(self) -> V1SampledDeviceResidentDLookReceipt:
        if self._look_receipt is None:
            raise RuntimeError("device-resident D packet has no closed receipt")
        return self._look_receipt

    def bind_route_chunk_size(self, role: str, chunk_size: int) -> None:
        if role not in self._ROLES or role in self._bound_roles:
            raise RuntimeError("device-resident D branch was bound out of order")
        if _positive_int(chunk_size, name="device-resident D route chunk size") != (
            self.registered_route_chunk_size
        ):
            raise ValueError(
                "device-resident D provider route chunk size differs from plan"
            )
        self._bound_roles.add(role)

    def _open_session_once(self) -> None:
        if self._session_packet_open:
            return
        packet = self.plan.reduced_request.packet_request
        stratum_name = self.plan.fd_request.stratum_labels[0]
        stratum_index = self.plan.fd_request.design.stratum_names.index(stratum_name)
        order_start = int(packet.sealed_order_start_indices[stratum_index])
        self.session.begin_registered_look_packet(
            registered_look_index=self.registered_look_index,
            packet_ordinal=self.packet_ordinal,
            context_ids=packet.expected_context_ids,
            draw_ids=packet.expected_draw_ids,
            stratum_labels=self.plan.fd_request.stratum_labels,
            sealed_order_indices=tuple(
                range(order_start, order_start + len(packet.expected_context_ids))
            ),
            source_measurement_hashes=self.plan.fd_request.source_measurement_hashes,
        )
        self._session_packet_open = True

    def begin_branch(
        self,
        role: str,
        *,
        context_ids: Sequence[str],
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        stream_identity_hash: str | None,
    ) -> None:
        if (
            role not in self._bound_roles
            or role in self._open_roles
            or role in self._closed_roles
        ):
            raise RuntimeError("device-resident D branch was not bound exactly once")
        packet = self.plan.reduced_request.packet_request
        if tuple(context_ids) != tuple(packet.expected_context_ids):
            raise ValueError("device-resident D contexts differ from packet plan")
        if tuple(route_ids) != self.route_ids:
            raise ValueError("device-resident D route axis differs from packet plan")
        if str(destination_transport_hash or "") != self.destination_transport_hash:
            raise ValueError("device-resident D destination transport differs")
        if str(stream_identity_hash or "") != self.stream_identities[role]:
            raise ValueError("device-resident D stream identity differs from plan")
        self._open_session_once()
        self._open_roles.add(role)

    def _same_capture_chunk_source_hash(self, *, start: int, stop: int) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_device_resident_D_same_capture_chunk_v1",
                "transaction_request_hash": self.transaction_request_hash,
                "packet_plan_hash": self.plan.plan_hash,
                "packet_ordinal": self.packet_ordinal,
                "route_window": (int(start), int(stop)),
                "corrected_evidence_stream_identity_hash": (
                    self.stream_identities["corrected_a"]
                ),
                "literal_jacobian_stream_identity_hash": (
                    self.stream_identities["literal_j"]
                ),
                "same_capture_and_vjp_required": True,
                "route_scale_values_not_hashed": True,
            },
            prefix=_DEVICE_RESIDENT_D_CHUNK_SOURCE_PREFIX,
        )

    def consume_branch(
        self,
        role: str,
        *,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        begin = int(start)
        end = int(stop)
        if role not in self._open_roles or role in self._closed_roles:
            raise RuntimeError("device-resident D branch has no open packet")
        expected = self._next_route_by_role[role]
        if (
            begin != expected
            or end <= begin
            or end > len(self.route_ids)
            or end - begin > self.registered_route_chunk_size
        ):
            raise ValueError("device-resident D chunks changed exact route order")
        window = (begin, end)
        if self._pending_rows is None:
            self._pending_role = role
            self._pending_window = window
            self._pending_rows = rows
        else:
            if self._pending_role == role or self._pending_window != window:
                raise RuntimeError("device-resident D chunks are not exact A/J pairs")
            if self._pending_rows is rows:
                raise RuntimeError(
                    "device-resident D corrected rows were reused as literal-J rows"
                )
            corrected = (
                self._pending_rows if self._pending_role == "corrected_a" else rows
            )
            literal = (
                self._pending_rows if self._pending_role == "literal_j" else rows
            )
            self.session.consume_same_capture_route_chunk(
                route_start=begin,
                corrected_a_rows=corrected,
                literal_j_rows=literal,
                route_axis_hash=self.route_axis_hash,
                corrected_a_stream_identity_hash=(
                    self.stream_identities["corrected_a"]
                ),
                literal_j_stream_identity_hash=self.stream_identities["literal_j"],
                same_capture_chunk_source_hash=self._same_capture_chunk_source_hash(
                    start=begin,
                    stop=end,
                ),
            )
            self._pending_role = None
            self._pending_window = None
            self._pending_rows = None
        self._next_route_by_role[role] = end
        self._windows_by_role[role].append(window)

    def end_branch(self, role: str) -> None:
        if (
            role not in self._open_roles
            or role in self._closed_roles
            or self._next_route_by_role[role] != len(self.route_ids)
        ):
            raise RuntimeError("device-resident D branch did not cover its route axis")
        self._closed_roles.add(role)
        if self._closed_roles == set(self._ROLES):
            if self._pending_rows is not None:
                raise RuntimeError("device-resident D ended with one unpaired chunk")
            self._look_receipt = self.session.end_registered_look_packet()

    def _branch_payload(self, role: str) -> dict[str, object]:
        if self._look_receipt is None or self._closed_roles != set(self._ROLES):
            raise RuntimeError("device-resident D branch receipt is premature")
        windows = tuple(self._windows_by_role[role])
        if windows != tuple(self._windows_by_role[self._ROLES[0]]):
            raise RuntimeError("device-resident D branch window ledgers diverged")
        return {
            "schema": "partition_free_v1_sampled_device_resident_D_branch_stream_v1",
            "transaction_request_hash": self.transaction_request_hash,
            "session_design_hash": self.session.design.design_hash,
            "hbm_forecast_hash": self.session.hbm_forecast.forecast_hash,
            "look_receipt_hash": self._look_receipt.receipt_hash,
            "packet_plan_hash": self.plan.plan_hash,
            "packet_ordinal": self.packet_ordinal,
            "role": role,
            "positive_removal_stream_coordinate": (
                RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            ),
            "positive_removal_stream_identity_hash": self.stream_identities[role],
            "route_windows": windows,
            "context_count": len(self.plan.fd_request.context_ids),
            "route_count": len(self.route_ids),
            "device_resident_d_session_consumed": True,
            "route_scale_d2h_bytes": 0,
            "route_scale_host_tensor_materializations": 0,
            "proposal_discovery_input_authority": False,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }

    def branch_receipt(self, role: str) -> Mapping[str, object]:
        payload = self._branch_payload(role)
        payload.update(
            {
                "typed_receipt_hash": content_hash(
                    payload,
                    prefix=_DEVICE_RESIDENT_D_BRANCH_RECEIPT_PREFIX,
                ),
                "route_chunked_streaming": True,
                "route_chunk_size": self.registered_route_chunk_size,
                "route_chunk_update_count": len(payload["route_windows"]),
                "loss_only_route_streaming": True,
                "sampled_device_resident_d_session_streaming": True,
                "direct_positive_removal_chunk_count": len(
                    payload["route_windows"]
                ),
                "full_tripartite_rows_materialized": False,
                "host_context_route_rows_transferred": False,
                "host_context_route_row_transfer_count": 0,
                "host_context_route_row_transfer_bytes": 0,
                "host_context_route_transfer_role": "none",
                "fused_route_moments_requested": False,
                "fused_route_moments_engaged": False,
                "fused_route_moment_update_count": 0,
                "fused_score_epilogue_receipts": (),
            }
        )
        return MappingProxyType(payload)


def _verify_device_resident_d_provider_receipt(
    receipt: Mapping[str, object],
    *,
    plan: V1SampledLookPacketPlan,
    bridge: _DeviceResidentDSessionPacketBridge,
) -> str:
    """Verify the provider's zero-publication dual stream for live D."""

    if not isinstance(receipt, Mapping):
        raise TypeError("device-resident D provider returned no typed receipt")
    packet = plan.reduced_request.packet_request
    expected_context_hash = content_hash(
        packet.expected_context_ids,
        prefix=_PROVIDER_CONTEXT_PREFIX,
    )
    expected_scalars = {
        "schema": "optimized_device_tripartite_panel_receipt_v1",
        "context_count": len(packet.expected_context_ids),
        "context_packet_count": 1,
        "context_ids_hash": expected_context_hash,
        "route_count": packet.route_count,
        "route_chunked_streaming": True,
        "loss_only_route_streaming": True,
        "dual_chart_streaming": True,
        "loss_only_stream_tee": False,
        "shared_positive_removal_rows": False,
        "same_capture_and_vjp_required": True,
        "corrected_evidence_stream_identity_hash": (
            plan.corrected_evidence_stream_identity_hash
        ),
        "literal_jacobian_stream_identity_hash": (
            plan.literal_jacobian_stream_identity_hash
        ),
        "positive_removal_stream_coordinate": (
            RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        ),
        "positive_removal_stream_identity_hash": (
            plan.corrected_evidence_stream_identity_hash
        ),
        "full_tripartite_rows_materialized": False,
        "host_context_route_rows_transferred": False,
        "host_context_route_row_transfer_count": 0,
        "host_context_route_row_transfer_bytes": 0,
        "host_context_route_transfer_role": "none",
    }
    for name, expected in expected_scalars.items():
        if receipt.get(name) != expected:
            raise ValueError(
                f"device-resident D provider receipt differs on {name}"
            )
    evidence = dict(bridge.branch_receipt("corrected_a"))
    jacobian = dict(bridge.branch_receipt("literal_j"))
    if (
        dict(receipt.get("dual_chart_evidence_receipt", {})).get(
            "typed_receipt_hash"
        )
        != evidence["typed_receipt_hash"]
        or dict(receipt.get("dual_chart_jacobian_receipt", {})).get(
            "typed_receipt_hash"
        )
        != jacobian["typed_receipt_hash"]
    ):
        raise ValueError("device-resident D provider child receipts do not bind")
    if (
        int(evidence.get("route_scale_d2h_bytes", 0)) != 0
        or int(jacobian.get("route_scale_d2h_bytes", 0)) != 0
        or int(evidence.get("route_scale_host_tensor_materializations", 0)) != 0
        or int(jacobian.get("route_scale_host_tensor_materializations", 0)) != 0
    ):
        raise ValueError("device-resident D branch receipt published route rows")

    raw_receipts = tuple(
        receipt.get("exact_conserving_target_logit_stream_receipts", ())
    )
    if len(raw_receipts) != 1 or not isinstance(raw_receipts[0], Mapping):
        raise ValueError("device-resident D requires one raw dual-stream receipt")
    raw = raw_receipts[0]
    raw_expected = {
        "schema": "exact_dual_chart_positive_removal_stream_v1",
        "context_count": len(packet.expected_context_ids),
        "capture_batch_count": 1,
        "vjp_count": 1,
        "route_count": packet.route_count,
        "stream_coordinate": "raw_target_logit",
        "stream_identity_hash": plan.corrected_evidence_stream_identity_hash,
        "raw_target_logit_unit_pullback": True,
        "dual_corrected_jacobian_streaming": True,
        "same_capture_and_vjp": True,
        "corrected_rows_excluded_from_response_geometry": True,
        "literal_jacobian_stream_identity_hash": (
            plan.literal_jacobian_stream_identity_hash
        ),
        "corrected_minus_literal_mismatch_diagnostic_only": True,
    }
    for name, expected in raw_expected.items():
        if raw.get(name) != expected:
            raise ValueError(
                f"device-resident D raw dual-stream receipt differs on {name}"
            )
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_device_resident_D_provider_proof_v1",
            "provider_schema": receipt["schema"],
            "packet_plan_hash": plan.plan_hash,
            "look_receipt_hash": bridge.look_receipt.receipt_hash,
            "evidence_child_receipt_hash": evidence["typed_receipt_hash"],
            "literal_jacobian_child_receipt_hash": jacobian["typed_receipt_hash"],
            "raw_dual_stream_receipt": dict(raw),
            "route_scale_d2h_bytes": 0,
            "route_scale_host_tensor_materializations": 0,
        },
        prefix=_DEVICE_RESIDENT_D_TRANSACTION_RESULT_PREFIX + "provider-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledLookAggregateScoreContributionReceipt:
    """Compact score-reducer child result for one physical packet.

    The packet's routewise corrected-evidence rows are not materialized as a
    standalone host packet.  Instead, they are merged immediately into one
    transaction-private device accumulator.  The authoritative value object is
    the final ``V1StratifiedCorrectedAttributionMoments`` state; this receipt
    records the exact pre-registered packet contribution that fed that state.
    """

    transaction_request_hash: str
    packet_plan_hash: str
    packet_request_hash: str
    packet_ordinal: int
    route_axis_hash: str
    context_axis_hash: str
    destination_transport_hash: str
    raw_logit_stream_identity_hash: str
    device_identity: str
    route_count: int
    packet_stratum_counts: tuple[int, int]
    sealed_order_start_indices: tuple[int, int]
    registered_route_chunk_size: int
    route_windows: tuple[tuple[int, int], ...]
    raw_target_logit_row_semantics: str | None
    look_aggregate_transition_kind: str = (
        "device_packet_level_chan_merge_into_registered_look_score_state_v1"
    )
    authority_scope: str = SAMPLED_DISCOVERY_AUTHORITY_SCOPE
    schema: str = SAMPLED_LOOK_AGGREGATE_SCORE_CONTRIBUTION_SCHEMA_V1
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "transaction_request_hash",
            "packet_plan_hash",
            "packet_request_hash",
            "route_axis_hash",
            "context_axis_hash",
            "destination_transport_hash",
            "raw_logit_stream_identity_hash",
            "device_identity",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        ordinal = _positive_int(self.packet_ordinal + 1, name="packet ordinal plus one")
        route_count = _positive_int(self.route_count, name="route count")
        chunk_size = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        counts = tuple(
            int(value)
            for value in self.packet_stratum_counts
            if not isinstance(value, bool)
        )
        starts = tuple(
            int(value)
            for value in self.sealed_order_start_indices
            if not isinstance(value, bool)
        )
        if len(counts) != 2 or any(value < 0 for value in counts) or sum(counts) <= 0:
            raise ValueError("aggregate score contribution requires one nonempty stratum")
        if len(starts) != 2 or any(value < 0 for value in starts):
            raise ValueError("aggregate score contribution starts are invalid")
        nonempty = tuple(index for index, count in enumerate(counts) if count > 0)
        if len(nonempty) != 1:
            raise ValueError("aggregate score contribution must be same-stratum")
        windows = tuple((int(start), int(stop)) for start, stop in self.route_windows)
        if not windows:
            raise ValueError("aggregate score contribution requires route windows")
        cursor = 0
        for start, stop in windows:
            if start != cursor or stop <= start or stop > route_count:
                raise ValueError("aggregate score contribution windows are not a prefix")
            if stop - start > chunk_size:
                raise ValueError("aggregate score contribution chunk exceeds plan")
            cursor = stop
        if cursor != route_count:
            raise ValueError("aggregate score contribution did not cover the route axis")
        if (
            self.raw_target_logit_row_semantics
            != CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
        ):
            raise ValueError(
                "aggregate score contribution requires corrected Aumann--Shapley rows"
            )
        if self.authority_scope != SAMPLED_DISCOVERY_AUTHORITY_SCOPE:
            raise ValueError("aggregate score contribution authority is invalid")
        if self.schema != SAMPLED_LOOK_AGGREGATE_SCORE_CONTRIBUTION_SCHEMA_V1:
            raise ValueError("aggregate score contribution schema is invalid")
        object.__setattr__(self, "packet_ordinal", ordinal - 1)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "packet_stratum_counts", counts)
        object.__setattr__(self, "sealed_order_start_indices", starts)
        object.__setattr__(self, "registered_route_chunk_size", chunk_size)
        object.__setattr__(self, "route_windows", windows)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "transaction_request_hash": self.transaction_request_hash,
            "packet_plan_hash": self.packet_plan_hash,
            "packet_request_hash": self.packet_request_hash,
            "packet_ordinal": self.packet_ordinal,
            "route_axis_hash": self.route_axis_hash,
            "context_axis_hash": self.context_axis_hash,
            "destination_transport_hash": self.destination_transport_hash,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "device_identity": self.device_identity,
            "route_count": self.route_count,
            "packet_stratum_counts": self.packet_stratum_counts,
            "sealed_order_start_indices": self.sealed_order_start_indices,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "route_windows": self.route_windows,
            "raw_target_logit_row_semantics": self.raw_target_logit_row_semantics,
            "look_aggregate_transition_kind": self.look_aggregate_transition_kind,
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        observed = content_hash(
            self._hash_payload(),
            prefix=_LOOK_SCORE_CONTRIBUTION_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("aggregate score contribution receipt was tampered")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def contribution_hash(self) -> str:
        return content_hash(
            {
                "schema": SAMPLED_LOOK_AGGREGATE_SCORE_RESULT_SCHEMA_V1,
                "contribution_receipt_hash": self.receipt_hash,
                "packet_request_hash": self.packet_request_hash,
                "packet_plan_hash": self.packet_plan_hash,
                "packet_ordinal": self.packet_ordinal,
                "route_axis_hash": self.route_axis_hash,
                "look_aggregate_transition_kind": self.look_aggregate_transition_kind,
            },
            prefix=_LOOK_SCORE_RESULT_PREFIX,
        )


class _DeviceRegisteredLookScoreAccumulator:
    """Transaction-private device accumulator for all corrected-evidence packets."""

    def __init__(
        self,
        request: V1SampledRegisteredLookTransactionRequest,
        *,
        device: Any,
    ) -> None:
        if torch is None:
            raise RuntimeError("look-level score aggregation requires PyTorch")
        self.request = request
        first_request = request.packet_plans[0].reduced_request.packet_request
        self.route_count = first_request.route_count
        self.device = _canonical_device(device)
        shape = (2, self.route_count)
        self._signed_mean = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._signed_m2 = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._absolute_mean = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._absolute_m2 = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.packet_counts = [0, 0]
        self.contexts: list[list[str]] = [[], []]
        self.draws: list[list[str]] = [[], []]
        self.packet_count = 0
        self.packet_merge_seconds = 0.0
        self.packet_input_bytes = 0
        self.peak_packet_input_bytes = 0
        self._open_packet: dict[str, object] | None = None
        self._closed = False
        self._contribution_receipts: list[
            V1SampledLookAggregateScoreContributionReceipt
        ] = []
        self._runtime_diagnostics: dict[str, float | int] = {
            "score_look_aggregate_device_mode": 1,
            "score_reduction_host_enqueue_seconds": 0.0,
            "score_reduction_chunk_count": 0,
            "score_reduction_input_bytes": 0,
            "score_d2h_synchronization_inclusive_seconds": 0.0,
            "score_d2h_array_count": 0,
            "score_d2h_bytes": 0,
            "score_canonicalization_seconds": 0.0,
            "score_canonicalization_array_count": 0,
            "score_canonicalization_bytes": 0,
            "score_explicit_full_array_digest_seconds": 0.0,
            "score_explicit_full_array_digest_call_count": 0,
            "score_explicit_full_array_digest_bytes": 0,
            "score_look_aggregate_packet_count": 0,
            "score_look_aggregate_final_host_transfer_bytes": 0,
            "score_look_aggregate_avoided_packet_host_transfer_bytes": 0,
        }

    @property
    def runtime_diagnostics(self) -> Mapping[str, float | int]:
        return MappingProxyType(dict(self._runtime_diagnostics))

    @property
    def contribution_receipts(
        self,
    ) -> tuple[V1SampledLookAggregateScoreContributionReceipt, ...]:
        return tuple(self._contribution_receipts)

    @property
    def aggregate_array_bytes(self) -> int:
        return int(4 * 2 * self.route_count * _LE_F64.itemsize)

    def begin_packet(
        self,
        *,
        packet_ordinal: int,
        plan: V1SampledLookPacketPlan,
    ) -> None:
        if self._closed or self._open_packet is not None:
            raise RuntimeError("look-level score aggregation packet state is invalid")
        if packet_ordinal != self.packet_count:
            raise ValueError("look-level score aggregation packets are out of order")
        if plan is not self.request.packet_plans[packet_ordinal]:
            raise ValueError("look-level score aggregation plan is outside order")
        packet = plan.reduced_request.packet_request
        nonempty = tuple(
            index for index, count in enumerate(packet.packet_stratum_counts) if count > 0
        )
        if len(nonempty) != 1:
            raise ValueError("look-level score aggregation requires same-stratum packets")
        stratum = nonempty[0]
        prior_counts = (
            (0, 0)
            if self.request.prior_corrected_attribution_state is None
            else self.request.prior_corrected_attribution_state.stratum_counts
        )
        expected_start = int(prior_counts[stratum]) + int(self.packet_counts[stratum])
        if packet.sealed_order_start_indices[stratum] != expected_start:
            raise ValueError("look-level score aggregation packet prefix is stale")
        self._open_packet = {
            "packet_ordinal": packet_ordinal,
            "plan": plan,
            "stratum": stratum,
            "chunk_count": int(packet.packet_stratum_counts[stratum]),
            "left_count": int(self.packet_counts[stratum]),
            "next_route": 0,
            "route_windows": [],
        }

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        if self._closed or self._open_packet is None:
            raise RuntimeError("look-level score aggregation has no open packet")
        begin = int(start)
        end = int(stop)
        if begin < 0 or end <= begin or end > self.route_count:
            raise ValueError("look-level score aggregation route window is invalid")
        next_route = int(self._open_packet["next_route"])
        if begin != next_route:
            raise ValueError("look-level score aggregation chunks are out of order")
        plan = self._open_packet["plan"]
        assert isinstance(plan, V1SampledLookPacketPlan)
        packet = plan.reduced_request.packet_request
        if end - begin > packet.registered_route_chunk_size:
            raise ValueError("look-level score aggregation chunk exceeds plan")
        if (
            not isinstance(rows, torch.Tensor)
            or tuple(rows.shape)
            != (int(self._open_packet["chunk_count"]), end - begin)
            or rows.dtype != torch.float64
            or _canonical_device(rows.device) != self.device
            or not rows.is_contiguous()
        ):
            raise ValueError(
                "look-level score chunk must be contiguous float64 [draw,route-window] "
                "on the declared device"
            )
        if not bool(torch.isfinite(rows).all().item()):
            raise ValueError("look-level score chunk contains non-finite values")
        values = rows.detach()
        started = time.perf_counter()
        self._merge_packet_window(begin=begin, end=end, values=values)
        self._runtime_diagnostics["score_reduction_host_enqueue_seconds"] = float(
            self._runtime_diagnostics["score_reduction_host_enqueue_seconds"]
        ) + (time.perf_counter() - started)
        self._runtime_diagnostics["score_reduction_chunk_count"] = int(
            self._runtime_diagnostics["score_reduction_chunk_count"]
        ) + 1
        chunk_bytes = int(values.numel()) * int(values.element_size())
        self._runtime_diagnostics["score_reduction_input_bytes"] = int(
            self._runtime_diagnostics["score_reduction_input_bytes"]
        ) + chunk_bytes
        self.packet_input_bytes += chunk_bytes
        self.peak_packet_input_bytes = max(self.peak_packet_input_bytes, chunk_bytes)
        route_windows = self._open_packet["route_windows"]
        assert isinstance(route_windows, list)
        route_windows.append((begin, end))
        self._open_packet["next_route"] = end

    def _packet_moments(self, values: Any) -> tuple[Any, Any, Any, Any]:
        width = int(values.shape[1])
        signed_mean = torch.zeros(width, dtype=torch.float64, device=self.device)
        signed_m2 = torch.zeros(width, dtype=torch.float64, device=self.device)
        absolute_mean = torch.zeros(width, dtype=torch.float64, device=self.device)
        absolute_m2 = torch.zeros(width, dtype=torch.float64, device=self.device)
        count = 0
        for row in values:
            count += 1
            signed_delta = row - signed_mean
            signed_mean.add_(signed_delta / count)
            signed_m2.add_(signed_delta * (row - signed_mean))
            absolute = torch.abs(row)
            absolute_delta = absolute - absolute_mean
            absolute_mean.add_(absolute_delta / count)
            absolute_m2.add_(absolute_delta * (absolute - absolute_mean))
        return signed_mean, signed_m2, absolute_mean, absolute_m2

    def _merge_packet_window(self, *, begin: int, end: int, values: Any) -> None:
        assert self._open_packet is not None
        stratum = int(self._open_packet["stratum"])
        chunk_count = int(self._open_packet["chunk_count"])
        left_count = int(self._open_packet["left_count"])
        total = left_count + chunk_count
        (
            signed_mean,
            signed_m2,
            absolute_mean,
            absolute_m2,
        ) = self._packet_moments(values)
        for destination_mean, destination_m2, source_mean, source_m2 in (
            (
                self._signed_mean[stratum, begin:end],
                self._signed_m2[stratum, begin:end],
                signed_mean,
                signed_m2,
            ),
            (
                self._absolute_mean[stratum, begin:end],
                self._absolute_m2[stratum, begin:end],
                absolute_mean,
                absolute_m2,
            ),
        ):
            if left_count == 0:
                destination_mean.copy_(source_mean)
                destination_m2.copy_(source_m2)
            else:
                delta = source_mean - destination_mean
                destination_mean.add_(delta * (chunk_count / total))
                destination_m2.add_(
                    source_m2
                    + delta.square() * (left_count * chunk_count / total)
                )

    def end_packet(self) -> V1SampledLookAggregateScoreContributionReceipt:
        if self._closed or self._open_packet is None:
            raise RuntimeError("look-level score aggregation has no open packet")
        plan = self._open_packet["plan"]
        assert isinstance(plan, V1SampledLookPacketPlan)
        packet = plan.reduced_request.packet_request
        if int(self._open_packet["next_route"]) != packet.route_count:
            raise RuntimeError("look-level score aggregation packet is partial")
        stratum = int(self._open_packet["stratum"])
        self.packet_counts[stratum] += int(self._open_packet["chunk_count"])
        self.contexts[stratum].extend(packet.expected_context_ids_by_stratum[stratum])
        self.draws[stratum].extend(packet.expected_draw_ids_by_stratum[stratum])
        windows = tuple(self._open_packet["route_windows"])
        context_axis_hash = content_hash(
            packet.expected_context_ids,
            prefix=_PROVIDER_CONTEXT_PREFIX,
        )
        receipt = V1SampledLookAggregateScoreContributionReceipt(
            transaction_request_hash=self.request.request_hash,
            packet_plan_hash=plan.plan_hash,
            packet_request_hash=packet.request_hash,
            packet_ordinal=int(self._open_packet["packet_ordinal"]),
            route_axis_hash=packet.route_axis_hash,
            context_axis_hash=context_axis_hash,
            destination_transport_hash=plan.reduced_request.destination_transport_hash,
            raw_logit_stream_identity_hash=(
                plan.reduced_request.raw_logit_stream_identity_hash
            ),
            device_identity=_device_identity(self.device),
            route_count=packet.route_count,
            packet_stratum_counts=packet.packet_stratum_counts,
            sealed_order_start_indices=packet.sealed_order_start_indices,
            registered_route_chunk_size=packet.registered_route_chunk_size,
            route_windows=windows,
            raw_target_logit_row_semantics=(
                plan.reduced_request.raw_target_logit_row_semantics
            ),
        )
        self._contribution_receipts.append(receipt)
        self.packet_count += 1
        self._runtime_diagnostics["score_look_aggregate_packet_count"] = (
            self.packet_count
        )
        self._open_packet = None
        return receipt

    def close_moments(self) -> V1StratifiedCorrectedAttributionMoments:
        if self._closed:
            raise RuntimeError("look-level score aggregation is already closed")
        if self._open_packet is not None or self.packet_count != len(
            self.request.packet_plans
        ):
            raise RuntimeError("look-level score aggregation packet ledger is partial")
        saturated_strata = _saturated_zero_packet_strata(self.request)
        if sum(self.packet_counts) <= 0 or any(
            count <= 0 and stratum not in saturated_strata
            for stratum, count in enumerate(self.packet_counts)
        ):
            raise ValueError(
                "look-level score aggregation did not advance every unsaturated stratum"
            )
        signed_mean = _copy_aggregate_to_host_array(
            self._signed_mean,
            role="look aggregate signed mean",
            route_count=self.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        signed_m2 = _copy_aggregate_to_host_array(
            self._signed_m2,
            role="look aggregate signed M2",
            route_count=self.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        absolute_mean = _copy_aggregate_to_host_array(
            self._absolute_mean,
            role="look aggregate absolute mean",
            route_count=self.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        absolute_m2 = _copy_aggregate_to_host_array(
            self._absolute_m2,
            role="look aggregate absolute M2",
            route_count=self.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        aggregate_bytes = int(
            signed_mean.nbytes
            + signed_m2.nbytes
            + absolute_mean.nbytes
            + absolute_m2.nbytes
        )
        self._runtime_diagnostics[
            "score_look_aggregate_final_host_transfer_bytes"
        ] = aggregate_bytes
        packet_count = max(1, self.packet_count)
        self._runtime_diagnostics[
            "score_look_aggregate_avoided_packet_host_transfer_bytes"
        ] = max(0, (packet_count - 1) * aggregate_bytes)
        prior = self.request.prior_corrected_attribution_state
        prior_counts = (0, 0) if prior is None else prior.stratum_counts
        first_request = self.request.packet_plans[0].reduced_request.packet_request
        reduced: dict[str, object] = {
            "sealed_order_start_indices": prior_counts,
            "packet_stratum_counts": tuple(self.packet_counts),
            "context_ids_by_stratum": tuple(tuple(values) for values in self.contexts),
            "draw_ids_by_stratum": tuple(tuple(values) for values in self.draws),
            "expected_context_ids_by_stratum": tuple(
                tuple(values) for values in self.contexts
            ),
            "expected_draw_ids_by_stratum": tuple(tuple(values) for values in self.draws),
            "signed_mean_by_stratum": signed_mean,
            "signed_m2_by_stratum": signed_m2,
            "absolute_mean_by_stratum": absolute_mean,
            "absolute_m2_by_stratum": absolute_m2,
        }
        if prior is not None:
            result = prior.append_reduced_stratum_moments(**reduced)
        else:
            result = V1StratifiedCorrectedAttributionMoments.from_reduced_stratum_moments(
                campaign_contract_hash=first_request.campaign_contract_hash,
                model_identity_hash=first_request.model_identity_hash,
                query_contract_hash=first_request.query_contract_hash,
                preprocessing_hash=first_request.preprocessing_hash,
                current_mask_hash=first_request.current_mask_hash,
                raw_logit_stream_source_hash=first_request.raw_logit_stream_source_hash,
                sampling_law_hash=first_request.sampling_law_hash,
                proposal_generation=first_request.proposal_generation,
                route_axis_hash=first_request.route_axis_hash,
                stratum_names=first_request.stratum_names,
                population_stratum_counts=first_request.population_stratum_counts,
                sealed_population_order_hashes=(
                    first_request.sealed_population_order_hashes
                ),
                **reduced,
            )
        if result.look_index != self.request.registered_look_index:
            raise RuntimeError("look-level score aggregation advanced another look")
        _assert_saturated_moment_carry(
            request=self.request,
            moments=result,
            saturated_strata=saturated_strata,
        )
        self._closed = True
        return result


class _LookAggregatingReducedPacketLossOnlyAdapter(
    _SampledStreamTransaction,
    _RouteChunkedLossOnlyConsumer,
):
    """Per-packet provider adapter backed by one look-level score accumulator."""

    positive_removal_stream_coordinate = RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE

    def __init__(
        self,
        request: V1SampledReducedPacketStreamRequest,
        *,
        packet_ordinal: int,
        look_accumulator: _DeviceRegisteredLookScoreAccumulator,
        dual_chunk_observer: Any | None = None,
    ) -> None:
        if type(request) is not V1SampledReducedPacketStreamRequest:
            raise TypeError("look aggregate score adapter requires exact request")
        request.assert_integrity()
        if request.schema != SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2:
            raise ValueError(
                "look aggregate score adapter is only canonical dual-chart v2"
            )
        if not callable(dual_chunk_observer):
            raise ValueError("look aggregate score adapter requires a dual observer")
        self.request = request
        self._packet_ordinal = packet_ordinal
        self._look_accumulator = look_accumulator
        self._dual_chunk_observer = dual_chunk_observer
        self.route_ids = request.route_ids
        self.device = _canonical_device(request.declared_device)
        self.loss_pullback_hash = request.raw_logit_stream_identity_hash
        self.destination_transport_hash = request.destination_transport_hash
        self.head_ids = ("positive_removal:raw_target_logit",)
        self.chunk_size = request.packet_request.registered_route_chunk_size
        self._route_chunk_bound = False
        self._packet_open = False
        self._closed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._next_route = 0
        self._route_windows: list[tuple[int, int]] = []
        self._input_tensor_object_ids: list[int] = []
        self._contribution_receipt: (
            V1SampledLookAggregateScoreContributionReceipt | None
        ) = None
        self._receipt: V1SampledReducedStreamReceipt | None = None
        self._closure: V1SampledProposalDiscoveryClosure | None = None
        self._runtime_diagnostics: dict[str, float | int] = {
            "score_adapter_child_close_synchronization_inclusive_seconds": 0.0,
            "score_adapter_commit_construction_seconds": 0.0,
            "score_adapter_look_aggregate_contribution_mode": 1,
        }

    @property
    def input_tensor_object_ids(self) -> tuple[int, ...]:
        return tuple(self._input_tensor_object_ids)

    @property
    def runtime_diagnostics(self) -> Mapping[str, float | int]:
        return MappingProxyType(dict(self._runtime_diagnostics))

    @property
    def contribution_receipt(self) -> V1SampledLookAggregateScoreContributionReceipt:
        if self._contribution_receipt is None:
            raise RuntimeError("look aggregate score contribution is unavailable")
        return self._contribution_receipt

    @property
    def contribution_hash(self) -> str:
        return self.contribution_receipt.contribution_hash

    @property
    def receipt(self) -> V1SampledReducedStreamReceipt:
        if self._receipt is None:
            raise RuntimeError("reduced stream receipt is unavailable before close")
        return self._receipt

    @property
    def closure(self) -> V1SampledProposalDiscoveryClosure:
        if self._closure is None:
            raise RuntimeError("reduced stream closure is unavailable before close")
        return self._closure

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        try:
            self._assert_live()
            observed = _positive_int(chunk_size, name="provider route chunk size")
            if observed != self.request.packet_request.registered_route_chunk_size:
                raise ValueError("provider route chunk size differs from frozen packet")
            if self._packet_open:
                raise RuntimeError("cannot bind route chunks after packet open")
            self.chunk_size = observed
            self._route_chunk_bound = True
        except Exception as error:
            self._abort(error)

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        try:
            self._assert_live()
            if not self._route_chunk_bound:
                raise RuntimeError("provider route chunk size was not bound")
            if self._packet_open:
                raise RuntimeError("sampled reduced packet is already open")
            contexts = _identity_axis(
                context_ids,
                name="observed look aggregate packet contexts",
                unique=True,
            )
            if contexts != self.request.packet_request.expected_context_ids:
                raise ValueError("reduced packet contexts differ from frozen request")
            if len(route_ids) != len(self.route_ids) or ordered_string_axis_hash(
                route_ids,
                name="look aggregate observed route axis",
            ) != ordered_string_axis_hash(
                self.route_ids,
                name="look aggregate expected route axis",
            ):
                raise ValueError("reduced packet route axis differs from frozen request")
            if str(destination_transport_hash or "") != self.destination_transport_hash:
                raise ValueError("reduced packet destination transport differs")
            if str(loss_pullback_hash or "") != self.loss_pullback_hash:
                raise ValueError("reduced packet raw-logit stream identity differs")
            plan = self._look_accumulator.request.packet_plans[self._packet_ordinal]
            self._look_accumulator.begin_packet(
                packet_ordinal=self._packet_ordinal,
                plan=plan,
            )
            self._packet_open = True
            self._next_route = 0
        except Exception as error:
            self._abort(error)

    def _validate_positive_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> tuple[int, int]:
        begin = int(start)
        end = int(stop)
        width = end - begin
        if (
            not self._packet_open
            or begin != self._next_route
            or width <= 0
            or width > self.chunk_size
            or end > self.request.packet_request.route_count
        ):
            raise ValueError("reduced chunks must follow the registered route order")
        return begin, end

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        try:
            self._assert_live()
            begin, end = self._validate_positive_chunk(start, stop, rows)
            self._dual_chunk_observer(
                "corrected_evidence",
                begin,
                end,
                rows,
            )
            self._input_tensor_object_ids.append(id(rows))
            self._look_accumulator.consume_positive_removal_chunk(begin, end, rows)
            self._route_windows.append((begin, end))
            self._next_route = end
        except Exception as error:
            self._abort(error)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._abort(
            PermissionError("sampled reduced adapter refuses generic/BCE rows")
        )

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        self._abort(
            PermissionError(
                "sampled reduced adapter refuses consumer-side pullback formation"
            )
        )

    def end_packet(self) -> None:
        try:
            self._assert_live()
            if (
                not self._packet_open
                or self._next_route != self.request.packet_request.route_count
            ):
                raise RuntimeError("reduced packet did not cover the route axis")
            child_close_started = time.perf_counter()
            contribution = self._look_accumulator.end_packet()
            self._runtime_diagnostics[
                "score_adapter_child_close_synchronization_inclusive_seconds"
            ] = float(
                self._runtime_diagnostics[
                    "score_adapter_child_close_synchronization_inclusive_seconds"
                ]
            ) + (time.perf_counter() - child_close_started)
            commit_started = time.perf_counter()
            receipt = V1SampledReducedStreamReceipt(
                request_hash=self.request.request_hash,
                child_request_hash=self.request.packet_request.request_hash,
                child_packet_hash=contribution.contribution_hash,
                child_derivation_receipt_hash=contribution.receipt_hash,
                route_axis_hash=self.request.packet_request.route_axis_hash,
                context_axis_hash=contribution.context_axis_hash,
                destination_transport_hash=self.destination_transport_hash,
                raw_logit_stream_identity_hash=self.loss_pullback_hash,
                device_identity=_device_identity(self.device),
                route_count=self.request.packet_request.route_count,
                registered_route_chunk_size=self.chunk_size,
                route_windows=tuple(self._route_windows),
                shared_tensor_observation_count=len(self._input_tensor_object_ids),
                raw_target_logit_row_semantics=(
                    self.request.raw_target_logit_row_semantics
                ),
                schema=SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA_V2,
            )
            closure = V1SampledProposalDiscoveryClosure(
                adapter_kind="gpu_reduced_corrected_attribution_packet",
                adapter_request_hash=self.request.request_hash,
                adapter_receipt_hash=receipt.receipt_hash,
                child_contract_hash=self.request.packet_request.request_hash,
                child_result_hash=contribution.contribution_hash,
                child_result_schema=SAMPLED_LOOK_AGGREGATE_SCORE_RESULT_SCHEMA_V1,
                raw_logit_stream_identity_hash=self.loss_pullback_hash,
            )
            self._runtime_diagnostics[
                "score_adapter_commit_construction_seconds"
            ] = float(
                self._runtime_diagnostics[
                    "score_adapter_commit_construction_seconds"
                ]
            ) + (time.perf_counter() - commit_started)
            self._contribution_receipt = contribution
            self._receipt = receipt
            self._closure = closure
            self._packet_open = False
            self._closed = True
        except Exception as error:
            self._abort(error)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        receipt = self.receipt
        closure = self.closure
        return MappingProxyType(
            {
                "schema": receipt.schema,
                "typed_receipt_hash": receipt.receipt_hash,
                "typed_discovery_closure_hash": closure.closure_hash,
                "route_chunked_streaming": True,
                "route_chunk_size": receipt.registered_route_chunk_size,
                "route_chunk_update_count": len(receipt.route_windows),
                "loss_only_route_streaming": True,
                "sampled_reduced_packet_streaming": True,
                "sampled_look_aggregate_score_streaming": True,
                "positive_removal_stream_coordinate": (
                    RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
                ),
                "positive_removal_stream_identity_hash": self.loss_pullback_hash,
                "direct_positive_removal_chunk_count": len(receipt.route_windows),
                "full_tripartite_rows_materialized": False,
                "host_context_route_rows_transferred": False,
                "host_context_route_row_transfer_count": 0,
                "host_context_route_row_transfer_bytes": 0,
                "host_context_route_transfer_role": "none",
                "raw_target_logit_row_semantics": (
                    self.request.raw_target_logit_row_semantics
                ),
                "proposal_discovery_input_authority": True,
                "fq_certification_authority": False,
                "bce_oracle_authority": False,
                "mask_commit_authority": False,
            }
        )


class _DualRowGroupBranchConsumer(_RouteChunkedLossOnlyConsumer):
    """One nominal branch of a paired, transaction-private row-shard sink."""

    positive_removal_stream_coordinate = RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE

    def __init__(
        self,
        *,
        bridge: _DualRowGroupPacketBridge,
        role: str,
        stream_identity_hash: str,
    ) -> None:
        if role not in {"corrected_a", "literal_j"}:
            raise ValueError("dual-row custody branch role is invalid")
        self._bridge = bridge
        self._role = role
        self.route_ids = bridge.route_ids
        self.device = bridge.device
        self.loss_pullback_hash = _identity(
            stream_identity_hash,
            name="dual-row custody stream identity hash",
        )
        self.head_ids = (f"{role}:raw_target_logit",)
        self.chunk_size = bridge.registered_route_chunk_size

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        self._bridge.bind_route_chunk_size(self._role, chunk_size)
        self.chunk_size = int(chunk_size)

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        self._bridge.begin_branch(
            self._role,
            context_ids=context_ids,
            route_ids=route_ids,
            destination_transport_hash=destination_transport_hash,
            stream_identity_hash=loss_pullback_hash,
        )

    def consume_positive_removal_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        self._bridge.consume_branch(self._role, start=start, stop=stop, rows=rows)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        raise PermissionError(
            "dual-row custody accepts only raw-logit positive-removal rows"
        )

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        raise PermissionError(
            "dual-row custody refuses consumer-side pullback formation"
        )

    def end_packet(self) -> None:
        self._bridge.end_branch(self._role)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        return self._bridge.branch_receipt(self._role)


class _DualRowGroupPacketBridge:
    """Pair distinct A/J branch chunks into one bounded group-stage writer.

    The provider-facing fan-out remains nominally split.  This bridge retains
    at most one unpaired route-window object, proves exact paired ordering,
    and coalesces the provider's source-native structural windows into the
    globally registered storage windows before forwarding a
    :class:`V1SampledDualChartRowChunk` to the private group transaction.  The
    coalescer owns at most one A/J storage-window pair.  This is required for
    real AlexNet execution: route chunks are naturally cut at layer/block
    boundaries, which need not coincide with the global storage grid.  It
    never publishes the group HEAD.
    """

    _ROLES: Final = ("corrected_a", "literal_j")

    def __init__(
        self,
        *,
        group_plan_hash: str,
        group_packet_plan: Any,
        staging_consumer: Any,
        route_ids: Sequence[str],
        device: Any,
        destination_transport_hash: str,
        registered_route_chunk_size: int,
        corrected_a_stream_identity_hash: str,
        literal_j_stream_identity_hash: str,
        dual_chunk_observer: Callable[[str, int, int, Any], None] | None = None,
    ) -> None:
        self.group_plan_hash = _identity(
            group_plan_hash,
            name="dual-row group plan hash",
        )
        self.group_packet_plan = group_packet_plan
        self.staging_consumer = staging_consumer
        self.route_ids = route_ids
        self.device = device
        self.destination_transport_hash = _identity(
            destination_transport_hash,
            name="dual-row destination transport hash",
        )
        self.registered_route_chunk_size = _positive_int(
            registered_route_chunk_size,
            name="dual-row registered route chunk size",
        )
        self.stream_identities = {
            "corrected_a": _identity(
                corrected_a_stream_identity_hash,
                name="corrected-A stream identity hash",
            ),
            "literal_j": _identity(
                literal_j_stream_identity_hash,
                name="literal-J stream identity hash",
            ),
        }
        if self.stream_identities["corrected_a"] == self.stream_identities[
            "literal_j"
        ]:
            raise ValueError("dual-row custody stream identities collapsed")
        if dual_chunk_observer is not None and not callable(dual_chunk_observer):
            raise TypeError("dual-row custody chunk observer must be callable")
        self._dual_chunk_observer = dual_chunk_observer
        self._bound_roles: set[str] = set()
        self._open_roles: set[str] = set()
        self._closed_roles: set[str] = set()
        self._next_route_by_role = {role: 0 for role in self._ROLES}
        self._windows_by_role = {role: [] for role in self._ROLES}
        self._pending_role: str | None = None
        self._pending_window: tuple[int, int] | None = None
        self._pending_rows: Any | None = None
        self._next_staged_route = 0
        self._assembly_corrected: Any | None = None
        self._assembly_literal: Any | None = None
        self._assembly_filled = 0
        self._sealed_result: Any | None = None

    @staticmethod
    def _allocate_like(rows: Any, *, shape: tuple[int, int]) -> Any:
        if isinstance(rows, np.ndarray):
            return np.empty(shape, dtype=rows.dtype, order="C")
        allocator = getattr(rows, "new_empty", None)
        if callable(allocator):
            return allocator(shape)
        raise TypeError(
            "dual-row structural coalescing requires NumPy or tensor-like rows"
        )

    @staticmethod
    def _copy_columns(
        destination: Any,
        *,
        destination_start: int,
        source: Any,
        source_start: int,
        width: int,
    ) -> None:
        destination_view = destination[
            :,
            destination_start : destination_start + width,
        ]
        source_view = source[:, source_start : source_start + width]
        if isinstance(destination, np.ndarray):
            np.copyto(destination_view, np.asarray(source_view), casting="no")
            return
        copier = getattr(destination_view, "copy_", None)
        if not callable(copier):
            raise TypeError("dual-row structural coalescer cannot copy row cells")
        copier(source_view)

    def _forward_paired_source_window(
        self,
        *,
        start: int,
        stop: int,
        corrected: Any,
        literal: Any,
    ) -> None:
        """Convert one exact provider-native A/J pair to storage-grid chunks."""

        context_count = len(self.group_packet_plan.context_ids)
        width = stop - start
        if (
            start != self._next_staged_route + self._assembly_filled
            or tuple(getattr(corrected, "shape", ())) != (context_count, width)
            or tuple(getattr(literal, "shape", ())) != (context_count, width)
        ):
            raise ValueError(
                "dual-row structural window differs from the exact paired cover"
            )

        source_offset = 0
        source_position = start
        route_count = len(self.route_ids)
        while source_position < stop:
            target_start = self._next_staged_route
            target_stop = min(
                route_count,
                target_start + self.registered_route_chunk_size,
            )
            target_width = target_stop - target_start

            # Preserve the zero-copy fast path whenever a provider-native
            # window already is one complete canonical storage window.
            if (
                self._assembly_corrected is None
                and self._assembly_literal is None
                and self._assembly_filled == 0
                and source_position == target_start
                and source_offset == 0
                and stop == target_stop
            ):
                self.staging_consumer.consume(
                    V1SampledDualChartRowChunk(
                        start=target_start,
                        stop=target_stop,
                        corrected_a_rows=corrected,
                        literal_j_rows=literal,
                    )
                )
                self._next_staged_route = target_stop
                source_position = stop
                source_offset = width
                continue

            if self._assembly_corrected is None:
                shape = (context_count, target_width)
                self._assembly_corrected = self._allocate_like(
                    corrected,
                    shape=shape,
                )
                self._assembly_literal = self._allocate_like(
                    literal,
                    shape=shape,
                )
                self._assembly_filled = 0

            available = target_width - self._assembly_filled
            take = min(stop - source_position, available)
            assert self._assembly_literal is not None
            self._copy_columns(
                self._assembly_corrected,
                destination_start=self._assembly_filled,
                source=corrected,
                source_start=source_offset,
                width=take,
            )
            self._copy_columns(
                self._assembly_literal,
                destination_start=self._assembly_filled,
                source=literal,
                source_start=source_offset,
                width=take,
            )
            self._assembly_filled += take
            source_position += take
            source_offset += take

            if self._assembly_filled == target_width:
                self.staging_consumer.consume(
                    V1SampledDualChartRowChunk(
                        start=target_start,
                        stop=target_stop,
                        corrected_a_rows=self._assembly_corrected,
                        literal_j_rows=self._assembly_literal,
                    )
                )
                self._next_staged_route = target_stop
                self._assembly_corrected = None
                self._assembly_literal = None
                self._assembly_filled = 0

        if source_offset != width:
            raise RuntimeError("dual-row structural coalescer lost source cells")

    @property
    def evidence_consumer(self) -> _DualRowGroupBranchConsumer:
        return _DualRowGroupBranchConsumer(
            bridge=self,
            role="corrected_a",
            stream_identity_hash=self.stream_identities["corrected_a"],
        )

    @property
    def jacobian_consumer(self) -> _DualRowGroupBranchConsumer:
        return _DualRowGroupBranchConsumer(
            bridge=self,
            role="literal_j",
            stream_identity_hash=self.stream_identities["literal_j"],
        )

    def bind_route_chunk_size(self, role: str, chunk_size: int) -> None:
        if role in self._bound_roles or self._open_roles or self._closed_roles:
            raise RuntimeError("dual-row custody branch was rebound")
        observed = _positive_int(chunk_size, name="dual-row provider chunk size")
        if observed != self.registered_route_chunk_size:
            raise ValueError("dual-row provider chunk size differs from group plan")
        self._bound_roles.add(role)

    def begin_branch(
        self,
        role: str,
        *,
        context_ids: Sequence[str],
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        stream_identity_hash: str | None,
    ) -> None:
        if role not in self._bound_roles or role in self._open_roles:
            raise RuntimeError("dual-row custody branch was not bound exactly once")
        if tuple(context_ids) != tuple(self.group_packet_plan.context_ids):
            raise ValueError("dual-row custody contexts differ from group packet")
        if len(route_ids) != len(self.route_ids) or ordered_string_axis_hash(
            route_ids,
            name="dual-row observed route axis",
        ) != ordered_string_axis_hash(
            self.route_ids,
            name="dual-row registered route axis",
        ):
            raise ValueError("dual-row custody route axis differs")
        if str(destination_transport_hash or "") != self.destination_transport_hash:
            raise ValueError("dual-row custody destination transport differs")
        if str(stream_identity_hash or "") != self.stream_identities[role]:
            raise ValueError("dual-row custody stream identity differs")
        self._open_roles.add(role)

    def consume_branch(
        self,
        role: str,
        *,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        begin = int(start)
        end = int(stop)
        if role not in self._open_roles or role in self._closed_roles:
            raise RuntimeError("dual-row custody branch has no open packet")
        expected = self._next_route_by_role[role]
        if (
            begin != expected
            or end <= begin
            or end > len(self.route_ids)
            or end - begin > self.registered_route_chunk_size
        ):
            raise ValueError("dual-row custody chunks changed exact route order")
        observer = self._dual_chunk_observer
        if observer is not None:
            observer(
                (
                    "corrected_evidence"
                    if role == "corrected_a"
                    else "literal_jacobian"
                ),
                begin,
                end,
                rows,
            )
        window = (begin, end)
        if self._pending_rows is None:
            self._pending_role = role
            self._pending_window = window
            self._pending_rows = rows
        else:
            if self._pending_role == role or self._pending_window != window:
                raise RuntimeError("dual-row custody chunks are not exact A/J pairs")
            if self._pending_rows is rows:
                raise RuntimeError("dual-row custody received one collapsed A/J object")
            corrected = self._pending_rows if self._pending_role == "corrected_a" else rows
            literal = self._pending_rows if self._pending_role == "literal_j" else rows
            self._forward_paired_source_window(
                start=begin,
                stop=end,
                corrected=corrected,
                literal=literal,
            )
            self._pending_role = None
            self._pending_window = None
            self._pending_rows = None
        self._next_route_by_role[role] = end
        self._windows_by_role[role].append(window)

    def end_branch(self, role: str) -> None:
        if (
            role not in self._open_roles
            or role in self._closed_roles
            or self._next_route_by_role[role] != len(self.route_ids)
        ):
            raise RuntimeError("dual-row custody branch did not cover its route axis")
        self._closed_roles.add(role)
        if self._closed_roles == set(self._ROLES):
            if self._pending_rows is not None:
                raise RuntimeError("dual-row custody ended with one unpaired chunk")
            if (
                self._next_staged_route != len(self.route_ids)
                or self._assembly_corrected is not None
                or self._assembly_literal is not None
                or self._assembly_filled != 0
            ):
                raise RuntimeError(
                    "dual-row structural coalescer ended with a partial storage window"
                )

    def _branch_payload(self, role: str) -> dict[str, object]:
        if self._closed_roles != set(self._ROLES) or self._pending_rows is not None:
            raise RuntimeError("dual-row custody branch receipt is premature")
        windows = tuple(self._windows_by_role[role])
        if windows != tuple(self._windows_by_role[self._ROLES[0]]):
            raise RuntimeError("dual-row custody branch window ledgers diverged")
        return {
            "schema": "partition_free_v1_sampled_dual_row_group_branch_stream_v1",
            "group_plan_hash": self.group_plan_hash,
            "group_packet_plan_hash": self.group_packet_plan.packet_plan_hash,
            "packet_ordinal": self.group_packet_plan.packet_ordinal,
            "role": role,
            "positive_removal_stream_coordinate": (
                RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            ),
            "positive_removal_stream_identity_hash": self.stream_identities[role],
            "route_windows": windows,
            "context_count": len(self.group_packet_plan.context_ids),
            "route_count": len(self.route_ids),
            "all_rows_staged_privately": True,
            "head_published": False,
            "proposal_discovery_input_authority": False,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }

    def branch_receipt(self, role: str) -> Mapping[str, object]:
        payload = self._branch_payload(role)
        payload.update(
            {
                "typed_receipt_hash": content_hash(
                    payload,
                    prefix=_DUAL_ROW_GROUP_BRANCH_RECEIPT_PREFIX,
                ),
                "route_chunked_streaming": True,
                "route_chunk_size": self.registered_route_chunk_size,
                "route_chunk_update_count": len(payload["route_windows"]),
                "loss_only_route_streaming": True,
                "sampled_dual_chart_row_group_staging": True,
                "direct_positive_removal_chunk_count": len(
                    payload["route_windows"]
                ),
                "full_tripartite_rows_materialized": False,
                "host_context_route_rows_transferred": True,
                "host_context_route_row_transfer_count": len(
                    payload["route_windows"]
                ),
                "host_context_route_row_transfer_bytes": (
                    len(self.group_packet_plan.context_ids)
                    * len(self.route_ids)
                    * _LE_F64.itemsize
                ),
                "host_context_route_transfer_role": f"immutable_{role}_row_custody",
            }
        )
        return MappingProxyType(payload)

    def seal(self) -> Any:
        if self._sealed_result is not None:
            raise RuntimeError("dual-row group packet staging was already sealed")
        if self._closed_roles != set(self._ROLES) or self._pending_rows is not None:
            raise RuntimeError("dual-row group packet staging is incomplete")
        self._sealed_result = self.staging_consumer.seal()
        return self._sealed_result


# Public names for the row-only transport primitives.  The historical
# underscored spellings remain the exact same classes so V5 replay identity is
# unchanged while source-bound adapters can reuse, rather than clone,
# their nominal provider ABI.
V1SampledDualChartChunkObjectSeparationWitness = (
    _DualChartChunkObjectSeparationWitness
)
V1SampledDualRowGroupBranchConsumer = _DualRowGroupBranchConsumer
V1SampledDualRowGroupPacketBridge = _DualRowGroupPacketBridge


def verify_v1_sampled_dual_chart_custody_only_provider_receipt(
    receipt: Mapping[str, object],
    *,
    group_plan: V1SampledDualChartRowGroupPlan,
    packet_ordinal: int,
    row_custody_bridge: _DualRowGroupPacketBridge,
    row_object_separation_receipt_hash: str,
) -> V1SampledDualChartCustodyProviderVerificationReceipt:
    """Verify one direct dual-chart provider call with no score/FD reducers.

    This is the row-custody-only sibling of :func:`_verify_provider_receipt`.
    It owns the same closed raw-stream/provider ABI, but the two direct child
    receipts are the exact custody branches rather than reducer-plus-custody
    tees.  It grants no statistical or downstream authority.
    """

    if not isinstance(receipt, Mapping):
        raise TypeError("dual-chart custody provider receipt must be a mapping")
    if type(group_plan) is not V1SampledDualChartRowGroupPlan:
        raise TypeError("dual-chart custody verification requires an exact group plan")
    group_plan.assert_integrity()
    ordinal = int(packet_ordinal)
    if (
        isinstance(packet_ordinal, bool)
        or ordinal < 0
        or ordinal >= len(group_plan.packet_plans)
    ):
        raise ValueError("dual-chart custody packet ordinal is outside the group plan")
    if type(row_custody_bridge) is not _DualRowGroupPacketBridge:
        raise TypeError("dual-chart custody verification requires the exact bridge")
    packet = group_plan.packet_plans[ordinal]
    packet.assert_integrity()
    if (
        row_custody_bridge.group_plan_hash != group_plan.group_plan_hash
        or row_custody_bridge.group_packet_plan is not packet
    ):
        raise ValueError("dual-chart custody bridge is stale or belongs to another packet")

    corrected_child = dict(row_custody_bridge.branch_receipt("corrected_a"))
    literal_child = dict(row_custody_bridge.branch_receipt("literal_j"))
    corrected_hash = _identity(
        corrected_child.get("typed_receipt_hash"),
        name="corrected-A custody branch receipt hash",
    )
    literal_hash = _identity(
        literal_child.get("typed_receipt_hash"),
        name="literal-J custody branch receipt hash",
    )
    separation_hash = _identity(
        row_object_separation_receipt_hash,
        name="dual-chart row-object separation receipt hash",
    )
    windows = tuple(corrected_child.get("route_windows", ()))
    if (
        not windows
        or tuple(literal_child.get("route_windows", ())) != windows
        or corrected_child.get("role") != "corrected_a"
        or literal_child.get("role") != "literal_j"
        or corrected_child.get("positive_removal_stream_identity_hash")
        != group_plan.corrected_a_stream_identity_hash
        or literal_child.get("positive_removal_stream_identity_hash")
        != group_plan.literal_j_stream_identity_hash
        or corrected_child.get("group_packet_plan_hash") != packet.packet_plan_hash
        or literal_child.get("group_packet_plan_hash") != packet.packet_plan_hash
    ):
        raise ValueError("direct custody branches are swapped, partial, or stale")

    expected_context_hash = content_hash(
        packet.context_ids,
        prefix=_PROVIDER_CONTEXT_PREFIX,
    )
    host_transfer_count = 2 * len(windows)
    host_transfer_bytes = (
        2 * len(packet.context_ids) * group_plan.route_count * _LE_F64.itemsize
    )
    top_exact = {
        "schema": "optimized_device_tripartite_panel_receipt_v1",
        "request_id": packet.provider_plan_identity_hash,
        "provider_plan_identity_hash": packet.provider_plan_identity_hash,
        "source_capture_identity_hash": packet.source_capture_identity_hash,
        "context_count": len(packet.context_ids),
        "context_packet_count": 1,
        "context_ids_hash": expected_context_hash,
        "route_count": group_plan.route_count,
        "route_chunked_streaming": True,
        "loss_only_route_streaming": True,
        "loss_only_stream_tee": False,
        "loss_only_stream_tee_child_count": 0,
        "shared_positive_removal_rows": False,
        "positive_removal_stream_coordinate": (
            RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        ),
        "positive_removal_stream_identity_hash": (
            group_plan.corrected_a_stream_identity_hash
        ),
        "full_tripartite_rows_materialized": False,
        "host_context_route_rows_transferred": True,
        "host_context_route_row_transfer_count": host_transfer_count,
        "host_context_route_row_transfer_bytes": host_transfer_bytes,
        "host_context_route_transfer_role": (
            "immutable_corrected_a_row_custody+immutable_literal_j_row_custody"
        ),
        "exact_conserving_single_pass_target_logit_streaming": True,
        "dual_chart_streaming": True,
        "dual_chart_child_count": 2,
        "same_capture_and_vjp_required": True,
        "corrected_evidence_stream_identity_hash": (
            group_plan.corrected_a_stream_identity_hash
        ),
        "literal_jacobian_stream_identity_hash": (
            group_plan.literal_j_stream_identity_hash
        ),
    }
    for name, expected in top_exact.items():
        if receipt.get(name) != expected:
            raise ValueError(
                f"direct dual-chart custody provider receipt {name} differs "
                "from preregistration"
            )
    if tuple(receipt.get("loss_only_stream_tee_child_receipts", ())) != ():
        raise ValueError("direct dual-chart custody provider introduced a branch tee")
    evidence_receipt = receipt.get("dual_chart_evidence_receipt")
    jacobian_receipt = receipt.get("dual_chart_jacobian_receipt")
    if (
        not isinstance(evidence_receipt, Mapping)
        or not isinstance(jacobian_receipt, Mapping)
        or dict(evidence_receipt) != corrected_child
        or dict(jacobian_receipt) != literal_child
    ):
        raise ValueError("direct dual-chart custody provider swapped or changed a child")

    raw_receipts = receipt.get("exact_conserving_target_logit_stream_receipts")
    if not isinstance(raw_receipts, (tuple, list)) or len(raw_receipts) != 1:
        raise ValueError("direct dual-chart custody requires one raw stream receipt")
    raw_receipt = raw_receipts[0]
    if not isinstance(raw_receipt, Mapping):
        raise TypeError("direct dual-chart custody raw stream receipt is not a mapping")
    _refuse_unregistered_v5_provider_claims(
        receipt,
        scope="sampled custody-only provider",
    )
    _refuse_unregistered_v5_provider_claims(
        raw_receipt,
        scope="sampled custody-only raw dual stream",
    )
    raw_exact = {
        "schema": "exact_dual_chart_positive_removal_stream_v1",
        "request_id": packet.provider_plan_identity_hash,
        "provider_plan_identity_hash": packet.provider_plan_identity_hash,
        "source_capture_identity_hash": packet.source_capture_identity_hash,
        "context_count": len(packet.context_ids),
        "capture_batch_count": 1,
        "vjp_count": 1,
        "route_count": group_plan.route_count,
        "route_chunk_size": group_plan.registered_route_chunk_size,
        "route_chunk_count": len(windows),
        "stream_coordinate": RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
        "stream_identity_hash": group_plan.corrected_a_stream_identity_hash,
        "positive_removal_orientation": True,
        "exact_conserving_attribution": True,
        "raw_target_logit_unit_pullback": True,
        "dual_corrected_jacobian_streaming": True,
        "same_capture_and_vjp": True,
        "corrected_rows_excluded_from_response_geometry": True,
        "literal_jacobian_stream_identity_hash": (
            group_plan.literal_j_stream_identity_hash
        ),
        "corrected_minus_literal_mismatch_diagnostic_only": True,
    }
    for name, expected in raw_exact.items():
        if raw_receipt.get(name) != expected:
            raise ValueError(
                f"direct dual-chart custody raw receipt {name} differs from "
                "preregistration"
            )
    entry_count = len(packet.context_ids) * group_plan.route_count
    mismatch_names = (
        "corrected_minus_literal_mismatch_mean_absolute",
        "corrected_minus_literal_mismatch_root_mean_square",
        "corrected_minus_literal_mismatch_linf",
        "corrected_minus_literal_mismatch_signed_mean",
    )
    mismatch = tuple(float(raw_receipt.get(name, float("nan"))) for name in mismatch_names)
    mismatch_nonzero = int(
        raw_receipt.get("corrected_minus_literal_mismatch_nonzero_count", -1)
    )
    if (
        int(raw_receipt.get("corrected_minus_literal_mismatch_entry_count", -1))
        != entry_count
        or mismatch_nonzero < 0
        or mismatch_nonzero > entry_count
        or not all(math.isfinite(value) for value in mismatch)
        or min(mismatch[:3]) < 0.0
    ):
        raise ValueError("direct dual-chart custody mismatch diagnostic is incomplete")

    raw_proof_payload = {
        "schema": "partition_free_v1_sampled_custody_only_raw_dual_stream_proof_v1",
        "registered_scientific_fields": raw_exact,
        "corrected_minus_literal_mismatch_entry_count": entry_count,
        "corrected_minus_literal_mismatch_nonzero_count": mismatch_nonzero,
        "corrected_minus_literal_mismatch_values": mismatch,
        "unregistered_provider_telemetry_excluded": True,
    }
    raw_proof_hash = content_hash(
        raw_proof_payload,
        prefix=_CUSTODY_PROVIDER_VERIFICATION_PREFIX + "raw-dual-stream-",
    )
    provider_proof_hash = content_hash(
        {
            "schema": (
                "partition_free_v1_sampled_custody_only_provider_normalized_"
                "proof_v1"
            ),
            "group_plan_hash": group_plan.group_plan_hash,
            "group_packet_plan_hash": packet.packet_plan_hash,
            "packet_ordinal": ordinal,
            "provider_plan_identity_hash": packet.provider_plan_identity_hash,
            "source_capture_identity_hash": packet.source_capture_identity_hash,
            "context_ids_hash": expected_context_hash,
            "route_count": group_plan.route_count,
            "route_window_count": len(windows),
            "raw_dual_stream_receipt_hash": raw_proof_hash,
            "row_object_separation_receipt_hash": separation_hash,
            "corrected_custody_branch_receipt_hash": corrected_hash,
            "literal_j_custody_branch_receipt_hash": literal_hash,
            "unregistered_privileged_provider_claims_refused": True,
        },
        prefix=_CUSTODY_PROVIDER_VERIFICATION_PREFIX + "normalized-",
    )
    return V1SampledDualChartCustodyProviderVerificationReceipt(
        group_plan_hash=group_plan.group_plan_hash,
        group_packet_plan_hash=packet.packet_plan_hash,
        packet_ordinal=ordinal,
        provider_plan_identity_hash=packet.provider_plan_identity_hash,
        source_capture_identity_hash=packet.source_capture_identity_hash,
        provider_receipt_hash=provider_proof_hash,
        raw_dual_stream_receipt_hash=raw_proof_hash,
        row_object_separation_receipt_hash=separation_hash,
        corrected_custody_branch_receipt_hash=corrected_hash,
        literal_j_custody_branch_receipt_hash=literal_hash,
        route_window_count=len(windows),
        _issuance_token=_CUSTODY_PROVIDER_VERIFICATION_ISSUER_TOKEN,
    )


def _verify_provider_receipt(
    receipt: Mapping[str, object],
    *,
    plan: V1SampledLookPacketPlan,
    score: V1SampledReducedPacketLossOnlyAdapter,
    fd: V1SampledFDLossOnlyAdapter,
    row_object_separation_receipt_hash: str | None = None,
    row_custody_bridge: _DualRowGroupPacketBridge | None = None,
) -> _VerifiedProviderPacket:
    """Close the exact legacy-tee or canonical dual-chart provider ABI."""

    packet = plan.reduced_request.packet_request
    expected_context_hash = content_hash(
        packet.expected_context_ids,
        prefix=_PROVIDER_CONTEXT_PREFIX,
    )
    dual_chart = plan.schema == SAMPLED_LOOK_PACKET_PLAN_SCHEMA_V2
    row_custody = row_custody_bridge is not None
    exact = {
        "schema": "optimized_device_tripartite_panel_receipt_v1",
        "context_count": len(packet.expected_context_ids),
        "context_packet_count": 1,
        "context_ids_hash": expected_context_hash,
        "route_count": packet.route_count,
        "route_chunked_streaming": True,
        "loss_only_route_streaming": True,
        "loss_only_stream_tee": not dual_chart,
        "shared_positive_removal_rows": not dual_chart,
        "positive_removal_stream_coordinate": (
            RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        ),
        "positive_removal_stream_identity_hash": (
            plan.reduced_request.raw_logit_stream_identity_hash
        ),
        "full_tripartite_rows_materialized": False,
        "host_context_route_rows_transferred": row_custody,
        "host_context_route_row_transfer_count": (
            2 * len(score.receipt.route_windows) if row_custody else 0
        ),
        "host_context_route_row_transfer_bytes": (
            2
            * len(packet.expected_context_ids)
            * packet.route_count
            * _LE_F64.itemsize
            if row_custody
            else 0
        ),
        "exact_conserving_single_pass_target_logit_streaming": True,
    }
    for name, expected in exact.items():
        if receipt.get(name) != expected:
            raise ValueError(
                f"sampled look provider receipt {name} differs from the frozen ABI"
            )
    provider_hash = content_hash(dict(receipt), prefix=_PROVIDER_PREFIX)
    if not dual_chart:
        if (
            row_object_separation_receipt_hash is not None
            or row_custody_bridge is not None
        ):
            raise ValueError("legacy sampled tee received a dual object witness")
        if receipt.get("loss_only_stream_tee_child_count") != 2:
            raise ValueError("sampled look legacy tee child count differs")
        children = receipt.get("loss_only_stream_tee_child_receipts")
        if not isinstance(children, (tuple, list)) or len(children) != 2:
            raise ValueError("sampled look provider receipt has a partial child ledger")
        expected_child_hashes = (
            score.receipt.receipt_hash,
            fd.receipt.receipt_hash,
        )
        observed_child_hashes = tuple(
            str(child.get("typed_receipt_hash", ""))
            if isinstance(child, Mapping)
            else ""
            for child in children
        )
        if observed_child_hashes != expected_child_hashes:
            raise ValueError(
                "sampled look provider receipt does not bind typed children"
            )
        if score.input_tensor_object_ids != fd.input_tensor_object_ids:
            raise RuntimeError(
                "sampled look tee did not offer identical chunk objects"
            )
        if not score.input_tensor_object_ids:
            raise RuntimeError("sampled look provider emitted no route chunks")
        return _VerifiedProviderPacket(provider_receipt_hash=provider_hash)

    evidence_identity = plan.corrected_evidence_stream_identity_hash
    jacobian_identity = plan.literal_jacobian_stream_identity_hash
    group_packet_plan = (
        None if row_custody_bridge is None else row_custody_bridge.group_packet_plan
    )
    evidence_child = receipt.get("dual_chart_evidence_receipt")
    jacobian_child = receipt.get("dual_chart_jacobian_receipt")
    if not isinstance(evidence_child, Mapping) or not isinstance(
        jacobian_child,
        Mapping,
    ):
        raise ValueError("sampled dual-chart provider receipt has a partial child ledger")
    raw_receipts = receipt.get("exact_conserving_target_logit_stream_receipts")
    if not isinstance(raw_receipts, (tuple, list)) or len(raw_receipts) != 1:
        raise ValueError("sampled dual chart requires one exact raw stream receipt")
    raw_receipt = raw_receipts[0]
    if not isinstance(raw_receipt, Mapping):
        raise TypeError("sampled dual raw stream receipt is not a mapping")
    if group_packet_plan is not None:
        # Provider-owned mappings are an extensible telemetry surface, not an
        # authority surface.  Reject privileged/path claims at those two
        # provider-owned levels, then hash only the closed scientific
        # projection below.  Reducer and custody children are validated by
        # their independently typed hashes/exact payloads.
        _refuse_unregistered_v5_provider_claims(
            receipt,
            scope="sampled v5 provider",
        )
        _refuse_unregistered_v5_provider_claims(
            raw_receipt,
            scope="sampled v5 raw dual stream",
        )
        expected_provider_plan = group_packet_plan.provider_plan_identity_hash
        expected_capture = group_packet_plan.source_capture_identity_hash
        if (
            receipt.get("request_id") != expected_provider_plan
            or receipt.get("provider_plan_identity_hash")
            != expected_provider_plan
            or receipt.get("source_capture_identity_hash") != expected_capture
        ):
            raise ValueError(
                "sampled v5 provider receipt does not bind the registered "
                "provider plan and source capture"
            )
        if (
            raw_receipt.get("provider_plan_identity_hash")
            != expected_provider_plan
            or raw_receipt.get("source_capture_identity_hash") != expected_capture
        ):
            raise ValueError(
                "sampled v5 raw dual stream receipt does not bind the "
                "registered provider plan and source capture"
            )
    route_window_count = len(score.receipt.route_windows)
    dual_exact = {
        "dual_chart_streaming": True,
        "dual_chart_child_count": 2,
        "same_capture_and_vjp_required": True,
        "corrected_evidence_stream_identity_hash": evidence_identity,
        "literal_jacobian_stream_identity_hash": jacobian_identity,
    }
    for name, expected in dual_exact.items():
        if receipt.get(name) != expected:
            raise ValueError(
                f"sampled dual-chart provider receipt {name} differs from the plan"
            )
    evidence_reducer_child: Mapping[str, object] = evidence_child
    jacobian_reducer_child: Mapping[str, object] = jacobian_child
    corrected_custody_child: Mapping[str, object] | None = None
    literal_j_custody_child: Mapping[str, object] | None = None
    if row_custody:
        evidence_nested = evidence_child.get("loss_only_stream_tee_child_receipts")
        jacobian_nested = jacobian_child.get("loss_only_stream_tee_child_receipts")
        if (
            evidence_child.get("loss_only_stream_tee") is not True
            or jacobian_child.get("loss_only_stream_tee") is not True
            or evidence_child.get("loss_only_stream_tee_child_count") != 2
            or jacobian_child.get("loss_only_stream_tee_child_count") != 2
            or not isinstance(evidence_nested, (tuple, list))
            or not isinstance(jacobian_nested, (tuple, list))
            or len(evidence_nested) != 2
            or len(jacobian_nested) != 2
            or not all(
                isinstance(value, Mapping)
                for value in (*evidence_nested, *jacobian_nested)
            )
        ):
            raise ValueError("sampled v5 provider has a partial branch fan-out")
        evidence_reducer_child = evidence_nested[0]
        corrected_custody_child = evidence_nested[1]
        jacobian_reducer_child = jacobian_nested[0]
        literal_j_custody_child = jacobian_nested[1]
        assert row_custody_bridge is not None
        expected_corrected_custody = dict(
            row_custody_bridge.branch_receipt("corrected_a")
        )
        expected_literal_custody = dict(
            row_custody_bridge.branch_receipt("literal_j")
        )
        if (
            dict(corrected_custody_child) != expected_corrected_custody
            or dict(literal_j_custody_child) != expected_literal_custody
        ):
            raise ValueError(
                "sampled v5 provider does not bind exact private custody children"
            )
    elif (
        evidence_child.get("loss_only_stream_tee") is True
        or jacobian_child.get("loss_only_stream_tee") is True
    ):
        raise ValueError("sampled v4 dual chart cannot claim v5 custody fan-out")
    if (
        tuple(receipt.get("loss_only_stream_tee_child_receipts", ())) != ()
        or receipt.get("loss_only_stream_tee_child_count") != 0
        or evidence_reducer_child.get("typed_receipt_hash")
        != score.receipt.receipt_hash
        or jacobian_reducer_child.get("typed_receipt_hash")
        != fd.receipt.receipt_hash
        or evidence_reducer_child.get("positive_removal_stream_identity_hash")
        != evidence_identity
        or jacobian_reducer_child.get("positive_removal_stream_identity_hash")
        != jacobian_identity
        or evidence_reducer_child.get("raw_target_logit_row_semantics")
        != CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
        or jacobian_reducer_child.get("raw_target_logit_row_semantics")
        != LITERAL_GATE_JACOBIAN_ROW_SEMANTICS
        or evidence_reducer_child.get("sampled_reduced_packet_streaming") is not True
        or jacobian_reducer_child.get("sampled_same_stratum_fd_streaming") is not True
        or int(
            evidence_reducer_child.get("direct_positive_removal_chunk_count", 0)
        )
        != route_window_count
        or int(
            jacobian_reducer_child.get("direct_positive_removal_chunk_count", 0)
        )
        != route_window_count
    ):
        raise ValueError(
            "sampled dual-chart provider receipt swapped, collapsed, or mistyped "
            "its typed children"
        )
    raw_exact = {
        "schema": "exact_dual_chart_positive_removal_stream_v1",
        "context_count": len(packet.expected_context_ids),
        "capture_batch_count": 1,
        "vjp_count": 1,
        "route_count": packet.route_count,
        "stream_coordinate": RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
        "stream_identity_hash": evidence_identity,
        "raw_target_logit_unit_pullback": True,
        "dual_corrected_jacobian_streaming": True,
        "same_capture_and_vjp": True,
        "corrected_rows_excluded_from_response_geometry": True,
        "literal_jacobian_stream_identity_hash": jacobian_identity,
        "corrected_minus_literal_mismatch_diagnostic_only": True,
    }
    if group_packet_plan is not None:
        raw_exact.update(
            {
                "provider_plan_identity_hash": (
                    group_packet_plan.provider_plan_identity_hash
                ),
                "source_capture_identity_hash": (
                    group_packet_plan.source_capture_identity_hash
                ),
                "route_chunk_size": packet.registered_route_chunk_size,
                "positive_removal_orientation": True,
                "exact_conserving_attribution": True,
            }
        )
    for name, expected in raw_exact.items():
        if raw_receipt.get(name) != expected:
            raise ValueError(
                f"sampled dual raw stream receipt {name} differs from the plan"
            )
    if (
        int(raw_receipt.get("route_chunk_count", 0)) != route_window_count
        or int(raw_receipt.get("corrected_minus_literal_mismatch_entry_count", 0))
        != len(packet.expected_context_ids) * packet.route_count
        or any(
            not math.isfinite(float(raw_receipt.get(name, float("nan"))))
            for name in (
                "corrected_minus_literal_mismatch_mean_absolute",
                "corrected_minus_literal_mismatch_root_mean_square",
                "corrected_minus_literal_mismatch_linf",
                "corrected_minus_literal_mismatch_signed_mean",
            )
        )
    ):
        raise ValueError("sampled dual raw stream receipt is incomplete")
    mismatch_fields = (
        "corrected_minus_literal_mismatch_mean_absolute",
        "corrected_minus_literal_mismatch_root_mean_square",
        "corrected_minus_literal_mismatch_linf",
        "corrected_minus_literal_mismatch_signed_mean",
    )
    mismatch_values = tuple(float(raw_receipt[name]) for name in mismatch_fields)
    mismatch_nonzero_count = int(
        raw_receipt.get("corrected_minus_literal_mismatch_nonzero_count", -1)
    )
    if group_packet_plan is not None and (
        mismatch_nonzero_count < 0
        or mismatch_nonzero_count
        > len(packet.expected_context_ids) * packet.route_count
        or min(mismatch_values[:3]) < 0.0
    ):
        raise ValueError("sampled v5 mismatch diagnostic is invalid")
    separation_hash = _identity(
        row_object_separation_receipt_hash,
        name="dual-chart row-object separation receipt hash",
    )
    raw_dual_stream_receipt_hash = (
        content_hash(
            dict(raw_receipt),
            prefix=_PROVIDER_PREFIX + "dual-stream-",
        )
        if group_packet_plan is None
        else content_hash(
            {
                "schema": (
                    "partition_free_v1_sampled_v5_normalized_raw_dual_"
                    "stream_proof_v1"
                ),
                "registered_scientific_fields": raw_exact,
                "route_chunk_count": route_window_count,
                "corrected_minus_literal_mismatch_entry_count": (
                    len(packet.expected_context_ids) * packet.route_count
                ),
                "corrected_minus_literal_mismatch_nonzero_count": (
                    mismatch_nonzero_count
                ),
                "corrected_minus_literal_mismatch_mean_absolute": (
                    mismatch_values[0]
                ),
                "corrected_minus_literal_mismatch_root_mean_square": (
                    mismatch_values[1]
                ),
                "corrected_minus_literal_mismatch_linf": mismatch_values[2],
                "corrected_minus_literal_mismatch_signed_mean": (
                    mismatch_values[3]
                ),
                # The per-route-block histogram is bound through the raw
                # receipt hash and verified by the push provider; it is NOT
                # mirrored here because this dict is a hashed legacy family.
                "unregistered_provider_telemetry_excluded": True,
            },
            prefix=_V5_PROVIDER_NORMALIZED_PROOF_PREFIX + "raw-dual-stream-",
        )
    )
    corrected_custody_branch_hash = (
        None
        if corrected_custody_child is None
        else _identity(
            corrected_custody_child.get("typed_receipt_hash"),
            name="corrected-A custody branch receipt hash",
        )
    )
    literal_j_custody_branch_hash = (
        None
        if literal_j_custody_child is None
        else _identity(
            literal_j_custody_child.get("typed_receipt_hash"),
            name="literal-J custody branch receipt hash",
        )
    )
    if group_packet_plan is not None:
        provider_hash = content_hash(
            {
                "schema": "partition_free_v1_sampled_v5_provider_normalized_proof_v1",
                "registered_packet_plan_hash": plan.plan_hash,
                "group_packet_plan_hash": group_packet_plan.packet_plan_hash,
                "packet_ordinal": group_packet_plan.packet_ordinal,
                "provider_plan_identity_hash": (
                    group_packet_plan.provider_plan_identity_hash
                ),
                "source_capture_identity_hash": (
                    group_packet_plan.source_capture_identity_hash
                ),
                "context_ids_hash": expected_context_hash,
                "route_count": packet.route_count,
                "route_window_count": route_window_count,
                "corrected_evidence_child_receipt_hash": (
                    score.receipt.receipt_hash
                ),
                "literal_jacobian_child_receipt_hash": fd.receipt.receipt_hash,
                "raw_dual_stream_receipt_hash": raw_dual_stream_receipt_hash,
                "row_object_separation_receipt_hash": separation_hash,
                "corrected_custody_branch_receipt_hash": (
                    corrected_custody_branch_hash
                ),
                "literal_j_custody_branch_receipt_hash": (
                    literal_j_custody_branch_hash
                ),
                "unregistered_privileged_provider_claims_refused": True,
            },
            prefix=_V5_PROVIDER_NORMALIZED_PROOF_PREFIX,
        )
    return _VerifiedProviderPacket(
        provider_receipt_hash=provider_hash,
        corrected_evidence_child_receipt_hash=score.receipt.receipt_hash,
        literal_jacobian_child_receipt_hash=fd.receipt.receipt_hash,
        raw_dual_stream_receipt_hash=raw_dual_stream_receipt_hash,
        row_object_separation_receipt_hash=separation_hash,
        corrected_custody_branch_receipt_hash=corrected_custody_branch_hash,
        literal_j_custody_branch_receipt_hash=literal_j_custody_branch_hash,
        capture_batch_count=1,
        vjp_count=1,
    )


def _saturated_zero_packet_strata(
    request: V1SampledRegisteredLookTransactionRequest,
) -> tuple[int, ...]:
    """Return strata omitted only because their registered census is closed."""

    observed = {
        plan.fd_request.stratum_labels[0] for plan in request.packet_plans
    }
    missing = tuple(
        index
        for index, name in enumerate(request.design.stratum_names)
        if name not in observed
    )
    if not missing:
        return ()
    prior = request.prior_corrected_attribution_state
    if prior is None:
        raise RuntimeError(
            "a zero-packet stratum cannot exist without a persisted census"
        )
    for index in missing:
        if not (
            prior.stratum_counts[index]
            == request.registered_look_cumulative_stratum_counts[index]
            == prior.population_stratum_counts[index]
        ):
            raise RuntimeError(
                "a zero-packet stratum lost its finite-population census proof"
            )
    return missing


def _assert_saturated_moment_carry(
    *,
    request: V1SampledRegisteredLookTransactionRequest,
    moments: V1StratifiedCorrectedAttributionMoments,
    saturated_strata: Sequence[int],
) -> None:
    """Fail closed unless omitted census strata survive the Chan merge exactly."""

    indices = tuple(saturated_strata)
    if not indices:
        return
    prior = request.prior_corrected_attribution_state
    if prior is None:
        raise RuntimeError("saturated-stratum moment carry lacks prior state")
    for index in indices:
        if (
            moments.stratum_counts[index] != prior.stratum_counts[index]
            or moments.observed_context_ids_by_stratum[index]
            != prior.observed_context_ids_by_stratum[index]
            or moments.observed_draw_ids_by_stratum[index]
            != prior.observed_draw_ids_by_stratum[index]
        ):
            raise RuntimeError(
                "a saturated zero-packet stratum changed its moment ledger"
            )
        for name in (
            "signed_mean_by_stratum",
            "signed_m2_by_stratum",
            "absolute_mean_by_stratum",
            "absolute_m2_by_stratum",
        ):
            current = np.asarray(getattr(moments, name)[index])
            persisted = np.asarray(getattr(prior, name)[index])
            if (
                current.dtype != persisted.dtype
                or current.shape != persisted.shape
                or current.tobytes(order="C") != persisted.tobytes(order="C")
            ):
                raise RuntimeError(
                    "a saturated zero-packet stratum changed its moment state"
                )


def _assert_saturated_fd_carry(
    *,
    request: V1SampledRegisteredLookTransactionRequest,
    snapshot: V1StratifiedFDAccumulatorSnapshot,
) -> None:
    """Require hash-identical FD children for omitted census strata."""

    indices = _saturated_zero_packet_strata(request)
    if not indices:
        return
    prior = request.prior_fd_snapshot
    if prior is None:
        raise RuntimeError("saturated-stratum FD carry lacks prior state")
    prior_children = dict(prior.stratum_snapshots)
    current_children = dict(snapshot.stratum_snapshots)
    for index in indices:
        name = request.design.stratum_names[index]
        if (
            current_children[name].snapshot_hash
            != prior_children[name].snapshot_hash
        ):
            raise RuntimeError(
                "a saturated zero-packet stratum changed its FD child state"
            )


def _merge_registered_look_packets(
    *,
    request: V1SampledRegisteredLookTransactionRequest,
    packets: Sequence[V1GPUReducedProposalPacket],
) -> tuple[V1StratifiedCorrectedAttributionMoments, float, float]:
    """Apply all physical stratum packets as one statistical-look transition.

    Same-stratum FD execution may require several bounded physical provider
    packets per stratum.  Those boundaries govern deterministic replay and
    rollback, not group-sequential alpha spending.  Coalescing the disjoint
    summaries here makes the complete all-strata transaction advance
    ``look_index`` once.
    """

    merge_started = time.perf_counter()
    closed = tuple(packets)
    plans = request.packet_plans
    if len(closed) != len(plans):
        raise ValueError("registered-look reduced packet ledger is partial")
    first_request = plans[0].reduced_request.packet_request
    route_count = first_request.route_count
    arrays = {
        "signed_mean_by_stratum": np.zeros((2, route_count), dtype=_LE_F64),
        "signed_m2_by_stratum": np.zeros((2, route_count), dtype=_LE_F64),
        "absolute_mean_by_stratum": np.zeros((2, route_count), dtype=_LE_F64),
        "absolute_m2_by_stratum": np.zeros((2, route_count), dtype=_LE_F64),
    }
    packet_counts = [0, 0]
    contexts: list[list[str]] = [[], []]
    draws: list[list[str]] = [[], []]
    prior = request.prior_corrected_attribution_state
    prior_counts = (0, 0) if prior is None else prior.stratum_counts
    for plan, packet in zip(plans, closed, strict=True):
        packet.assert_integrity(trust_cached_immutable=True)
        packet_request = packet.request
        if (
            packet_request.request_hash
            != plan.reduced_request.packet_request.request_hash
        ):
            raise ValueError("registered-look packet differs from its plan")
        nonempty = tuple(
            index
            for index, count in enumerate(packet_request.packet_stratum_counts)
            if count > 0
        )
        if len(nonempty) != 1:
            raise ValueError("registered-look physical packets must be same-stratum")
        stratum = nonempty[0]
        chunk_count = packet_request.packet_stratum_counts[stratum]
        left_count = packet_counts[stratum]
        total = left_count + chunk_count
        contexts[stratum].extend(
            packet_request.expected_context_ids_by_stratum[stratum]
        )
        draws[stratum].extend(packet_request.expected_draw_ids_by_stratum[stratum])
        for mean_name, m2_name in (
            ("signed_mean_by_stratum", "signed_m2_by_stratum"),
            ("absolute_mean_by_stratum", "absolute_m2_by_stratum"),
        ):
            destination_mean = arrays[mean_name]
            destination_m2 = arrays[m2_name]
            source_mean = getattr(packet, mean_name)[stratum]
            source_m2 = getattr(packet, m2_name)[stratum]
            if left_count == 0:
                destination_mean[stratum] = source_mean
                destination_m2[stratum] = source_m2
            else:
                delta = source_mean - destination_mean[stratum]
                destination_mean[stratum] += delta * (chunk_count / total)
                destination_m2[stratum] += (
                    source_m2
                    + np.square(delta, dtype=_LE_F64)
                    * (left_count * chunk_count / total)
                )
        packet_counts[stratum] = total
    saturated_strata = _saturated_zero_packet_strata(request)
    if sum(packet_counts) <= 0 or any(
        count <= 0 and stratum not in saturated_strata
        for stratum, count in enumerate(packet_counts)
    ):
        raise ValueError(
            "registered look did not advance every unsaturated stratum"
        )
    reduced: dict[str, object] = {
        "sealed_order_start_indices": prior_counts,
        "packet_stratum_counts": tuple(packet_counts),
        "context_ids_by_stratum": tuple(tuple(values) for values in contexts),
        "draw_ids_by_stratum": tuple(tuple(values) for values in draws),
        "expected_context_ids_by_stratum": tuple(
            tuple(values) for values in contexts
        ),
        "expected_draw_ids_by_stratum": tuple(tuple(values) for values in draws),
        **arrays,
    }
    merge_seconds = time.perf_counter() - merge_started
    canonicalization_started = time.perf_counter()
    if prior is not None:
        result = prior.append_reduced_stratum_moments(**reduced)
    else:
        result = V1StratifiedCorrectedAttributionMoments.from_reduced_stratum_moments(
            campaign_contract_hash=first_request.campaign_contract_hash,
            model_identity_hash=first_request.model_identity_hash,
            query_contract_hash=first_request.query_contract_hash,
            preprocessing_hash=first_request.preprocessing_hash,
            current_mask_hash=first_request.current_mask_hash,
            raw_logit_stream_source_hash=first_request.raw_logit_stream_source_hash,
            sampling_law_hash=first_request.sampling_law_hash,
            proposal_generation=first_request.proposal_generation,
            route_axis_hash=first_request.route_axis_hash,
            stratum_names=first_request.stratum_names,
            population_stratum_counts=first_request.population_stratum_counts,
            sealed_population_order_hashes=(
                first_request.sealed_population_order_hashes
            ),
            **reduced,
        )
    if result.look_index != request.registered_look_index:
        raise RuntimeError(
            "registered-look moments advanced by another statistical index"
        )
    _assert_saturated_moment_carry(
        request=request,
        moments=result,
        saturated_strata=saturated_strata,
    )
    canonicalization_seconds = time.perf_counter() - canonicalization_started
    return result, merge_seconds, canonicalization_seconds


class _StreamingRegisteredLookPacketMerger:
    """Merge verified packets while retaining one route-scale packet at most."""

    def __init__(self, request: V1SampledRegisteredLookTransactionRequest) -> None:
        self.request = request
        first_request = request.packet_plans[0].reduced_request.packet_request
        self.route_count = first_request.route_count
        self.arrays = {
            "signed_mean_by_stratum": np.zeros((2, self.route_count), dtype=_LE_F64),
            "signed_m2_by_stratum": np.zeros((2, self.route_count), dtype=_LE_F64),
            "absolute_mean_by_stratum": np.zeros((2, self.route_count), dtype=_LE_F64),
            "absolute_m2_by_stratum": np.zeros((2, self.route_count), dtype=_LE_F64),
        }
        self.packet_counts = [0, 0]
        self.contexts: list[list[str]] = [[], []]
        self.draws: list[list[str]] = [[], []]
        self.packet_input_bytes = 0
        self.peak_packet_input_bytes = 0
        self.packet_merge_seconds = 0.0
        self.packet_count = 0
        self._closed = False

    @property
    def aggregate_array_bytes(self) -> int:
        return sum(int(array.nbytes) for array in self.arrays.values())

    def add(
        self,
        *,
        plan: V1SampledLookPacketPlan,
        packet: V1GPUReducedProposalPacket,
    ) -> str:
        merge_started = time.perf_counter()
        if self._closed:
            raise RuntimeError("bounded-memory sampled look merger is closed")
        if self.packet_count >= len(self.request.packet_plans):
            raise ValueError("bounded-memory sampled look has an extra packet")
        expected_plan = self.request.packet_plans[self.packet_count]
        if plan is not expected_plan:
            raise ValueError(
                "bounded-memory sampled look packet is outside registered order"
            )
        packet_hash = packet.assert_integrity(trust_cached_immutable=True)
        packet_request = packet.request
        if (
            packet_request.request_hash
            != plan.reduced_request.packet_request.request_hash
        ):
            raise ValueError("registered-look packet differs from its plan")
        nonempty = tuple(
            index
            for index, count in enumerate(packet_request.packet_stratum_counts)
            if count > 0
        )
        if len(nonempty) != 1:
            raise ValueError("registered-look physical packets must be same-stratum")
        stratum = nonempty[0]
        chunk_count = packet_request.packet_stratum_counts[stratum]
        left_count = self.packet_counts[stratum]
        total = left_count + chunk_count
        self.contexts[stratum].extend(
            packet_request.expected_context_ids_by_stratum[stratum]
        )
        self.draws[stratum].extend(packet_request.expected_draw_ids_by_stratum[stratum])
        for mean_name, m2_name in (
            ("signed_mean_by_stratum", "signed_m2_by_stratum"),
            ("absolute_mean_by_stratum", "absolute_m2_by_stratum"),
        ):
            destination_mean = self.arrays[mean_name]
            destination_m2 = self.arrays[m2_name]
            source_mean = getattr(packet, mean_name)[stratum]
            source_m2 = getattr(packet, m2_name)[stratum]
            if left_count == 0:
                destination_mean[stratum] = source_mean
                destination_m2[stratum] = source_m2
            else:
                delta = source_mean - destination_mean[stratum]
                destination_mean[stratum] += delta * (chunk_count / total)
                destination_m2[stratum] += (
                    source_m2
                    + np.square(delta, dtype=_LE_F64)
                    * (left_count * chunk_count / total)
                )
        self.packet_counts[stratum] = total
        packet_input_bytes = sum(
            int(array.nbytes)
            for array in (
                packet.signed_mean_by_stratum,
                packet.signed_m2_by_stratum,
                packet.absolute_mean_by_stratum,
                packet.absolute_m2_by_stratum,
            )
        )
        self.packet_input_bytes += packet_input_bytes
        self.peak_packet_input_bytes = max(
            self.peak_packet_input_bytes,
            packet_input_bytes,
        )
        self.packet_count += 1
        self.packet_merge_seconds += time.perf_counter() - merge_started
        return packet_hash

    def close(self) -> V1StratifiedCorrectedAttributionMoments:
        if self._closed:
            raise RuntimeError("bounded-memory sampled look merger is already closed")
        if self.packet_count != len(self.request.packet_plans):
            raise ValueError("bounded-memory registered-look packet ledger is partial")
        saturated_strata = _saturated_zero_packet_strata(self.request)
        if sum(self.packet_counts) <= 0 or any(
            count <= 0 and stratum not in saturated_strata
            for stratum, count in enumerate(self.packet_counts)
        ):
            raise ValueError(
                "registered look did not advance every unsaturated stratum"
            )
        first_request = self.request.packet_plans[0].reduced_request.packet_request
        prior = self.request.prior_corrected_attribution_state
        prior_counts = (0, 0) if prior is None else prior.stratum_counts
        reduced: dict[str, object] = {
            "sealed_order_start_indices": prior_counts,
            "packet_stratum_counts": tuple(self.packet_counts),
            "context_ids_by_stratum": tuple(tuple(values) for values in self.contexts),
            "draw_ids_by_stratum": tuple(tuple(values) for values in self.draws),
            "expected_context_ids_by_stratum": tuple(
                tuple(values) for values in self.contexts
            ),
            "expected_draw_ids_by_stratum": (
                tuple(tuple(values) for values in self.draws)
            ),
            **self.arrays,
        }
        if prior is not None:
            result = prior.append_reduced_stratum_moments(**reduced)
        else:
            result = (
                V1StratifiedCorrectedAttributionMoments.from_reduced_stratum_moments(
                    campaign_contract_hash=first_request.campaign_contract_hash,
                    model_identity_hash=first_request.model_identity_hash,
                    query_contract_hash=first_request.query_contract_hash,
                    preprocessing_hash=first_request.preprocessing_hash,
                    current_mask_hash=first_request.current_mask_hash,
                    raw_logit_stream_source_hash=(
                        first_request.raw_logit_stream_source_hash
                    ),
                    sampling_law_hash=first_request.sampling_law_hash,
                    proposal_generation=first_request.proposal_generation,
                    route_axis_hash=first_request.route_axis_hash,
                    stratum_names=first_request.stratum_names,
                    population_stratum_counts=(
                        first_request.population_stratum_counts
                    ),
                    sealed_population_order_hashes=(
                        first_request.sealed_population_order_hashes
                    ),
                    **reduced,
                )
            )
        if result.look_index != self.request.registered_look_index:
            raise RuntimeError(
                "registered-look moments advanced by another statistical index"
            )
        _assert_saturated_moment_carry(
            request=self.request,
            moments=result,
            saturated_strata=saturated_strata,
        )
        self._closed = True
        return result


def _merge_bounded_registered_look_packet(
    *,
    merger: _StreamingRegisteredLookPacketMerger,
    plan: V1SampledLookPacketPlan,
    packet: V1GPUReducedProposalPacket,
) -> str:
    """Narrow testable joint for the one-packet bounded-memory merge."""

    return merger.add(plan=plan, packet=packet)


@dataclass(frozen=True, slots=True)
class V1SampledRegisteredLookPreparedCommitV5:
    """Unpublished outer V5 commit awaiting a durable backend binding.

    ``commit`` is fully validated, path-free, and pickleable.  This wrapper is
    intentionally process-local because it retains the publication lock and
    private group transaction.  The owning transaction must either publish it
    by exact outer commit hash or abort it.
    """

    commit: V1SampledRegisteredLookCommit
    runtime_diagnostics: Mapping[str, object]
    _group_transaction: V1SampledDualChartRowGroupTransaction = field(
        repr=False,
        compare=False,
    )
    _owner: Any = field(default=None, repr=False, compare=False)
    head_published: bool = False
    backend_durable_binding_required_before_head: bool = True
    _released: bool = field(default=False, repr=False, compare=False)
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _PREPARED_V5_ISSUANCE_TOKEN:
            raise PermissionError("prepared V5 commits are transaction-issued only")
        if (
            type(self.commit) is not V1SampledRegisteredLookCommit
            or self.commit.request.schema
            != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5
            or type(self._group_transaction)
            is not V1SampledDualChartRowGroupTransaction
            or type(self._owner) is not V1SampledRegisteredLookTransaction
        ):
            raise TypeError("prepared V5 wrapper has the wrong commit/group/owner type")
        self.commit.commit_hash
        custody = self.commit.dual_chart_row_custody_receipt
        if (
            custody is None
            or custody.group_commit_receipt.receipt_hash
            != self._group_transaction.prepare_commit().receipt.receipt_hash
            or self.head_published is not False
            or self.backend_durable_binding_required_before_head is not True
            or self._released is not False
        ):
            raise ValueError("prepared V5 wrapper differs from its unpublished group")
        object.__setattr__(
            self,
            "runtime_diagnostics",
            MappingProxyType(dict(self.runtime_diagnostics)),
        )

    @property
    def outer_commit_hash(self) -> str:
        return self.commit.commit_hash

    @property
    def row_group_commit_receipt(self) -> V1SampledDualChartRowGroupCommitReceipt:
        """Return the immutable path-free group receipt for crash journaling."""

        custody = self.commit.dual_chart_row_custody_receipt
        assert custody is not None
        prepared = self._group_transaction.prepare_commit()
        if prepared.receipt.receipt_hash != custody.group_commit_receipt.receipt_hash:
            raise RuntimeError("prepared V5 row-group receipt changed after closure")
        return custody.group_commit_receipt

    @property
    def prospective_row_prefix_manifest(
        self,
    ) -> V1SampledDualChartRowPrefixManifest:
        """Return the immutable path-free manifest that HEAD would publish."""

        prepared = self._group_transaction.prepare_commit()
        manifest = prepared.attachment.manifest
        if manifest.manifest_hash != self.row_group_commit_receipt.manifest_hash:
            raise RuntimeError("prepared V5 prospective manifest changed after closure")
        return manifest

    def close(self) -> None:
        """Idempotently abort unpublished custody owned by this live scope."""

        if self._released:
            return
        owner = self._owner
        if getattr(owner, "_prepared_v5", None) is self:
            owner.abort_prepared_v5()
            return
        if not (self.head_published or owner._closed or owner._failed):
            raise RuntimeError("prepared V5 ownership changed without release")
        object.__setattr__(self, "_released", True)

    def abort(self) -> None:
        """Explicit idempotent alias for scoped cleanup."""

        self.close()

    def __enter__(self) -> "V1SampledRegisteredLookPreparedCommitV5":
        if self._released or getattr(self._owner, "_prepared_v5", None) is not self:
            raise RuntimeError("prepared V5 scope is no longer live")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def __reduce__(self) -> Any:
        raise TypeError("prepared V5 wrappers retain a live publication lock")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError("prepared V5 wrappers retain a live publication lock")


class V1SampledRegisteredLookTransaction:
    """Execute a complete registered look against private rollback state."""

    def __init__(self, request: V1SampledRegisteredLookTransactionRequest) -> None:
        if type(request) is not V1SampledRegisteredLookTransactionRequest:
            raise TypeError("sampled look transaction requires its exact request")
        request.request_hash
        self.request = request
        self._failed = False
        self._failure_reason: str | None = None
        self._closed = False
        self._result: V1SampledRegisteredLookCommit | None = None
        self._runtime_diagnostics: Mapping[str, object] | None = None
        self._dual_chart_row_prefix_attachment: (
            V1SampledDualChartRowLivePrefixAttachment | None
        ) = None
        self._prepared_v5: V1SampledRegisteredLookPreparedCommitV5 | None = None

    @property
    def result(self) -> V1SampledRegisteredLookCommit:
        if self._result is None:
            raise RuntimeError("sampled look has no published commit")
        return self._result

    @property
    def is_failed(self) -> bool:
        return self._failed

    @property
    def runtime_diagnostics(self) -> Mapping[str, object]:
        """Return immutable physical metrics only after an atomic commit."""

        if self._runtime_diagnostics is None:
            raise RuntimeError("sampled look runtime diagnostics are unavailable")
        return self._runtime_diagnostics

    @property
    def dual_chart_row_prefix_attachment(
        self,
    ) -> V1SampledDualChartRowLivePrefixAttachment:
        """Return live path custody only after the V5 group HEAD publishes."""

        if self._dual_chart_row_prefix_attachment is None:
            raise RuntimeError("sampled look has no published dual-row attachment")
        return self._dual_chart_row_prefix_attachment

    def _fresh_private_accumulator(self) -> V1StratifiedDeviceFDTrustAccumulator:
        first = self.request.packet_plans[0].fd_request
        if self.request.prior_fd_snapshot is None:
            return V1StratifiedDeviceFDTrustAccumulator(
                self.request.design,
                sketch_rank=self.request.sketch_rank,
                device=first.declared_device,
                emit_derivation_receipt=False,
            )
        return V1StratifiedDeviceFDTrustAccumulator.from_snapshot(
            self.request.prior_fd_snapshot,
            design=self.request.design,
            sketch_rank=self.request.sketch_rank,
            device=first.declared_device,
        )

    def prepare_execute_v5(
        self,
        *,
        provider: Any,
        measurement_requests: Sequence[Any],
        dual_chart_row_scratch_directory: Any,
        prior_dual_chart_row_prefix: (
            V1SampledDualChartRowLivePrefixAttachment | None
        ) = None,
        aggregate_reduced_score_on_device: bool = False,
    ) -> V1SampledRegisteredLookPreparedCommitV5:
        """Close V5 science and files while preserving the predecessor HEAD."""

        if self.request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5:
            raise ValueError("prepared execution is defined only for schema V5")
        if type(aggregate_reduced_score_on_device) is not bool:
            raise TypeError("look-level score aggregation flag must be bool")
        result = self.execute(
            provider=provider,
            measurement_requests=measurement_requests,
            retain_reduced_packets=False,
            dual_chart_row_scratch_directory=dual_chart_row_scratch_directory,
            prior_dual_chart_row_prefix=prior_dual_chart_row_prefix,
            aggregate_reduced_score_on_device=aggregate_reduced_score_on_device,
            _prepare_v5=True,
        )
        if type(result) is not V1SampledRegisteredLookPreparedCommitV5:
            raise RuntimeError("sampled V5 preparation returned another boundary")
        return result

    def publish_prepared_v5(
        self,
        *,
        expected_outer_commit_hash: str,
    ) -> V1SampledRegisteredLookCommit:
        """Publish the one row HEAD after backend durable binding succeeds."""

        prepared = self._prepared_v5
        if prepared is None:
            raise RuntimeError("sampled V5 transaction has no prepared outer commit")
        try:
            expected = _identity(
                expected_outer_commit_hash,
                name="expected prepared V5 outer commit hash",
            )
            if expected != prepared.outer_commit_hash:
                raise ValueError("prepared V5 outer commit hash differs")
            custody = prepared.commit.dual_chart_row_custody_receipt
            assert custody is not None
            published = prepared._group_transaction.publish_prepared(
                expected_receipt_hash=custody.group_commit_receipt.receipt_hash
            )
        except Exception as error:
            # A failed publication protocol is terminal for this transaction.
            # In particular, a stale/wrong backend binding must not leave the
            # group lock or unpublished content-addressed files live and then
            # permit a second caller to guess another authorization hash.
            try:
                prepared._group_transaction.abort()
            finally:
                object.__setattr__(prepared, "_released", True)
                self._prepared_v5 = None
                self._runtime_diagnostics = None
                self._dual_chart_row_prefix_attachment = None
                self._failed = True
                self._failure_reason = (
                    "prepared V5 publication refused: "
                    f"{type(error).__name__}: {error}"
                )
            raise
        # No scientific construction follows the atomic HEAD replace.  Only
        # the already-validated in-process objects become visible here.
        self._result = prepared.commit
        self._runtime_diagnostics = prepared.runtime_diagnostics
        self._dual_chart_row_prefix_attachment = published.attachment
        object.__setattr__(prepared, "head_published", True)
        object.__setattr__(prepared, "_released", True)
        self._prepared_v5 = None
        self._closed = True
        return self._result

    def execute_into_device_resident_d_session(
        self,
        *,
        provider: Any,
        measurement_requests: Sequence[Any],
        device_resident_d_session: V1SampledDeviceResidentDSession,
    ) -> V1SampledDeviceResidentDTransactionResult:
        """Execute this registered D look into one live device-resident session.

        This path is deliberately narrower than :meth:`execute`: it has no
        reduced-packet publication, no private FD snapshot, no row-custody HEAD,
        and no commit authority.  It exists so the production D loop can keep a
        single live session across registered looks and defer all route-scale
        materialization until a decision-stable physical rank has been selected.
        """

        if type(device_resident_d_session) is not V1SampledDeviceResidentDSession:
            raise TypeError("device-resident D execution requires exact live session")
        if self.request.schema not in (
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
        ):
            raise ValueError(
                "device-resident D execution requires canonical dual-chart packets"
            )
        if self._failed:
            raise RuntimeError(
                "sampled look transaction is permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed or self._prepared_v5 is not None:
            raise RuntimeError("sampled look transaction is already closed")
        requests = tuple(measurement_requests)
        if len(requests) != len(self.request.packet_plans):
            raise ValueError("sampled D look measurement-request ledger is partial")
        execute = getattr(provider, "accumulate_device_tripartite", None)
        if not callable(execute):
            raise TypeError("sampled D provider lacks device tripartite execution")
        first = self.request.packet_plans[0]
        packet = first.reduced_request.packet_request
        session_design = device_resident_d_session.design
        session_coordinates = {
            "FD design": (
                session_design.fd_design.design_hash,
                self.request.design.design_hash,
            ),
            "route axis": (session_design.route_axis_hash, packet.route_axis_hash),
            "route count": (session_design.route_count, packet.route_count),
            "corrected-A stream": (
                session_design.corrected_a_stream_identity_hash,
                first.corrected_evidence_stream_identity_hash,
            ),
            "literal-J stream": (
                session_design.literal_j_stream_identity_hash,
                first.literal_jacobian_stream_identity_hash,
            ),
            "population counts": (
                session_design.population_stratum_counts,
                packet.population_stratum_counts,
            ),
            "population orders": (
                session_design.sealed_population_order_hashes,
                packet.sealed_population_order_hashes,
            ),
            "route chunk": (
                session_design.registered_route_chunk_size,
                packet.registered_route_chunk_size,
            ),
        }
        for name, (left, right) in session_coordinates.items():
            if left != right:
                raise ValueError(
                    f"device-resident D session differs from request on {name}"
                )
        prior_live = self.request.prior_device_resident_result
        existing_receipt_hashes = tuple(
            receipt.assert_integrity()
            for receipt in device_resident_d_session.look_receipts
        )
        if prior_live is None:
            if (
                self.request.registered_look_index != 1
                or existing_receipt_hashes
                or any(device_resident_d_session.stratum_counts)
            ):
                raise ValueError(
                    "first device-resident D request requires one empty live session"
                )
        else:
            if (
                prior_live.session_design_hash != session_design.design_hash
                or prior_live.hbm_forecast_hash
                != device_resident_d_session.hbm_forecast.forecast_hash
                or prior_live.live_session_custody_hash
                != content_hash(
                    {
                        "session_design_hash": session_design.design_hash,
                        "hbm_forecast_hash": (
                            device_resident_d_session.hbm_forecast.forecast_hash
                        ),
                    },
                    prefix=(
                        "partition-free-v1-sampled-device-resident-D-live-custody-"
                    ),
                )
                or existing_receipt_hashes
                != prior_live.cumulative_look_receipt_hashes
                or len(existing_receipt_hashes)
                != prior_live.cumulative_packet_count_after
                or device_resident_d_session.stratum_counts
                != prior_live.cumulative_stratum_counts_after
            ):
                raise ValueError(
                    "device-resident D continuation does not retain its exact live-session custody"
                )

        look_receipts: list[V1SampledDeviceResidentDLookReceipt] = []
        provider_hashes: list[str] = []
        provider_call_seconds: list[float] = []
        transaction_started = time.perf_counter()
        try:
            for packet_ordinal, (plan, measurement_request) in enumerate(
                zip(self.request.packet_plans, requests, strict=True)
            ):
                bridge = _DeviceResidentDSessionPacketBridge(
                    transaction_request_hash=self.request.request_hash,
                    registered_look_index=self.request.registered_look_index,
                    packet_ordinal=packet_ordinal,
                    plan=plan,
                    session=device_resident_d_session,
                )
                tee = build_route_chunked_dual_chart_fanout(
                    evidence_consumers=(bridge.evidence_consumer,),
                    jacobian_consumers=(bridge.jacobian_consumer,),
                )
                provider_started = time.perf_counter()
                provider_receipt = execute(measurement_request, accumulator=tee)
                provider_call_seconds.append(time.perf_counter() - provider_started)
                provider_hashes.append(
                    _verify_device_resident_d_provider_receipt(
                        provider_receipt,
                        plan=plan,
                        bridge=bridge,
                    )
                )
                look_receipts.append(bridge.look_receipt)
                del tee
                del provider_receipt
                del bridge
            result = V1SampledDeviceResidentDTransactionResult(
                transaction_request_hash=self.request.request_hash,
                session_design_hash=device_resident_d_session.design.design_hash,
                hbm_forecast_hash=(
                    device_resident_d_session.hbm_forecast.forecast_hash
                ),
                registered_look_index=self.request.registered_look_index,
                look_receipts=tuple(look_receipts),
                provider_receipt_hashes=tuple(provider_hashes),
                prior_result_hash=(
                    None if prior_live is None else prior_live.result_hash
                ),
                cumulative_packet_count_after=len(
                    device_resident_d_session.look_receipts
                ),
                cumulative_stratum_counts_after=(
                    device_resident_d_session.stratum_counts
                ),
                cumulative_look_receipt_hashes=tuple(
                    receipt.assert_integrity()
                    for receipt in device_resident_d_session.look_receipts
                ),
                live_session_custody_hash=content_hash(
                    {
                        "session_design_hash": session_design.design_hash,
                        "hbm_forecast_hash": (
                            device_resident_d_session.hbm_forecast.forecast_hash
                        ),
                    },
                    prefix=(
                        "partition-free-v1-sampled-device-resident-D-live-custody-"
                    ),
                ),
            )
            diagnostics: dict[str, object] = {
                "device_resident_d_transaction_seconds": (
                    time.perf_counter() - transaction_started
                ),
                "provider_vjp_call_synchronization_inclusive_seconds_by_packet": (
                    tuple(provider_call_seconds)
                ),
                "provider_vjp_call_synchronization_inclusive_seconds": sum(
                    provider_call_seconds
                ),
                "provider_vjp_call_count": len(provider_call_seconds),
                "registered_transport_packet_count": len(self.request.packet_plans),
                "route_scale_d2h_bytes": 0,
                "route_scale_host_tensor_materializations": 0,
                "device_resident_d_provider_peak_workspace_reserved_bytes": (
                    device_resident_d_session.design.provider_peak_workspace_bytes
                ),
                "device_resident_d_intermediate_scalar_receipt_count": len(
                    look_receipts
                ),
            }
            for name, raw_value in diagnostics.items():
                values = raw_value if isinstance(raw_value, tuple) else (raw_value,)
                for value in values:
                    if isinstance(value, bool) or not isinstance(
                        value, (int, float, np.integer, np.floating)
                    ):
                        raise TypeError(
                            f"sampled D runtime diagnostic {name} is not numeric"
                        )
                    if not math.isfinite(float(value)) or float(value) < 0.0:
                        raise ValueError(
                            f"sampled D runtime diagnostic {name} is invalid"
                        )
            self._runtime_diagnostics = MappingProxyType(dict(diagnostics))
            self._dual_chart_row_prefix_attachment = None
            self._closed = True
            return result
        except Exception as error:
            look_receipts.clear()
            provider_hashes.clear()
            # A provider or receipt failure can leave one device accumulator
            # partially advanced.  There is no inverse FD update.  Destroy the
            # live session so restart can only reconstruct it by deterministic
            # replay of the sealed prefix.
            device_resident_d_session.evict()
            self._runtime_diagnostics = None
            self._dual_chart_row_prefix_attachment = None
            self._failed = True
            self._failure_reason = f"{type(error).__name__}: {error}"
            raise

    def execute_into_device_resident_d_session_with_row_custody(
        self,
        *,
        provider: Any,
        measurement_requests: Sequence[Any],
        device_resident_d_session: V1SampledDeviceResidentDSession,
        row_scratch_directory: Any,
        prior_boundary: Any | None = None,
        prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None = None,
    ) -> Any:
        """Atomically tee one V6 look into live D and exact A/J custody.

        The implementation lives in the versioned V7 bridge to keep the
        historical V4 zero-publication and V5 portable transaction semantics
        byte-for-byte reachable.  A failure permanently refuses this
        transaction and destroys the non-invertible live D session.
        """

        if self._failed:
            raise RuntimeError(
                "sampled look transaction is permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed or self._prepared_v5 is not None:
            raise RuntimeError("sampled look transaction is already closed")
        if self.request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6:
            raise ValueError(
                "resident-plus-row execution requires the sampled V6 request"
            )
        try:
            from .modern_v1_sampled_v7_device_resident_d_bridge import (
                execute_v1_sampled_v7_device_resident_d_row_look,
            )

            result = execute_v1_sampled_v7_device_resident_d_row_look(
                request=self.request,
                provider=provider,
                measurement_requests=measurement_requests,
                session=device_resident_d_session,
                row_scratch_directory=row_scratch_directory,
                prior_boundary=prior_boundary,
                prior_row_prefix=prior_row_prefix,
            )
            self._runtime_diagnostics = result.runtime_diagnostics
            self._dual_chart_row_prefix_attachment = result.row_prefix_attachment
            self._closed = True
            return result
        except BaseException as error:
            # Even failures that occur during preflight must invalidate the
            # non-invertible session.  This makes every V6 refusal one atomic
            # abort/restart boundary instead of leaving a live owner that a
            # caller could accidentally reuse after a lineage mismatch.
            if type(device_resident_d_session) is V1SampledDeviceResidentDSession:
                device_resident_d_session.evict()
            self._runtime_diagnostics = None
            self._dual_chart_row_prefix_attachment = None
            self._failed = True
            self._failure_reason = f"{type(error).__name__}: {error}"
            raise

    def abort_prepared_v5(self) -> None:
        """Drop unpublished row files after backend durable binding refuses."""

        prepared = self._prepared_v5
        if prepared is None:
            raise RuntimeError("sampled V5 transaction has no prepared commit to abort")
        try:
            prepared._group_transaction.abort()
        finally:
            object.__setattr__(prepared, "_released", True)
            self._prepared_v5 = None
            self._runtime_diagnostics = None
            self._dual_chart_row_prefix_attachment = None
            self._failed = True
            self._failure_reason = "prepared V5 outer commit was explicitly aborted"

    def execute(
        self,
        *,
        provider: Any,
        measurement_requests: Sequence[Any],
        retain_reduced_packets: bool | None = None,
        dual_chart_row_scratch_directory: Any | None = None,
        prior_dual_chart_row_prefix: (
            V1SampledDualChartRowLivePrefixAttachment | None
        ) = None,
        aggregate_reduced_score_on_device: bool = False,
        _prepare_v5: bool = False,
    ) -> V1SampledRegisteredLookCommit | V1SampledRegisteredLookPreparedCommitV5:
        """Execute all stratum packets and return the sole publishable state."""

        if self.request.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6:
            raise PermissionError(
                "sampled V6 must use the atomic device-resident-D plus exact-row "
                "custody execution joint"
            )

        bounded_memory = (
            self.request.schema
            in (
                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3,
                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
                SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
            )
        )
        dual_chart = self.request.schema in (
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
            SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5,
        )
        row_custody = (
            self.request.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5
        )
        if type(_prepare_v5) is not bool:
            raise TypeError("private V5 preparation flag must be bool")
        if type(aggregate_reduced_score_on_device) is not bool:
            raise TypeError("look-level score aggregation flag must be bool")
        if aggregate_reduced_score_on_device and not (bounded_memory and dual_chart):
            raise ValueError(
                "look-level score aggregation is restricted to bounded dual-chart "
                "sampled transactions"
            )
        if row_custody and not _prepare_v5:
            raise PermissionError(
                "canonical V5 must use prepare_execute_v5, durably bind the outer "
                "commit, then call publish_prepared_v5"
            )
        if _prepare_v5 and not row_custody:
            raise ValueError("private V5 preparation cannot relabel V1-V4")
        if retain_reduced_packets is None:
            retain_reduced_packets = not bounded_memory
        elif type(retain_reduced_packets) is not bool:
            raise TypeError("reduced-packet retention flag must be bool or None")
        if retain_reduced_packets is bounded_memory:
            required_mode = (
                "hash-ledger bounded memory"
                if bounded_memory
                else "materialized packet custody"
            )
            raise ValueError(
                "sampled look request schema requires " + required_mode
            )
        if self._failed:
            raise RuntimeError(
                "sampled look transaction is permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed:
            raise RuntimeError("sampled look transaction is already closed")
        if self._prepared_v5 is not None:
            raise RuntimeError("sampled V5 transaction already has a prepared commit")
        requests = tuple(measurement_requests)
        if len(requests) != len(self.request.packet_plans):
            raise ValueError("sampled look measurement-request ledger is partial")
        execute = getattr(provider, "accumulate_device_tripartite", None)
        if not callable(execute):
            raise TypeError("sampled look provider lacks device tripartite execution")
        if row_custody:
            if dual_chart_row_scratch_directory is None:
                raise ValueError("sampled v5 execution requires node-local row scratch")
            assert self.request.dual_chart_row_group_plan is not None
            expected_provider_plans = tuple(
                packet.provider_plan_identity_hash
                for packet in self.request.dual_chart_row_group_plan.packet_plans
            )
            expected_capture_plans = tuple(
                packet.source_capture_identity_hash
                for packet in self.request.dual_chart_row_group_plan.packet_plans
            )
            if tuple(_provider_plan_identity_hash(value) for value in requests) != (
                expected_provider_plans
            ):
                raise ValueError(
                    "sampled v5 measurement requests differ from preregistered "
                    "provider plans"
                )
            if tuple(_source_capture_identity_hash(value) for value in requests) != (
                expected_capture_plans
            ):
                raise ValueError(
                    "sampled v5 measurement requests differ from preregistered "
                    "source-capture plans"
                )
        elif (
            dual_chart_row_scratch_directory is not None
            or prior_dual_chart_row_prefix is not None
        ):
            raise ValueError("sampled v1-v4 execution cannot publish v5 row custody")

        private_state: V1StratifiedDeviceFDTrustAccumulator | None = None
        packets: list[V1GPUReducedProposalPacket] = []
        packet_hashes: list[str] = []
        score_closures: list[V1SampledProposalDiscoveryClosure] = []
        append_results: list[V1SampledFDAppendResult] = []
        fd_closures: list[V1SampledProposalDiscoveryClosure] = []
        provider_hashes: list[str] = []
        evidence_child_receipt_hashes: list[str] = []
        jacobian_child_receipt_hashes: list[str] = []
        raw_dual_receipt_hashes: list[str] = []
        row_object_separation_receipt_hashes: list[str] = []
        corrected_custody_branch_receipt_hashes: list[str] = []
        literal_j_custody_branch_receipt_hashes: list[str] = []
        dual_capture_count = 0
        dual_vjp_count = 0
        diagnostics: dict[str, object] = {}
        provider_call_seconds: list[float] = []
        streaming_merger: _StreamingRegisteredLookPacketMerger | None = None
        score_look_accumulator: _DeviceRegisteredLookScoreAccumulator | None = None
        row_group_transaction: V1SampledDualChartRowGroupTransaction | None = None
        transaction_started = time.perf_counter()
        try:
            if row_custody:
                assert self.request.dual_chart_row_group_plan is not None
                row_group_transaction = (
                    begin_v1_sampled_dual_chart_row_group_transaction(
                        plan=self.request.dual_chart_row_group_plan,
                        scratch_directory=dual_chart_row_scratch_directory,
                        prior=prior_dual_chart_row_prefix,
                    )
                )
            restore_started = time.perf_counter()
            private_state = self._fresh_private_accumulator()
            # Snapshot restoration validates device state through existing
            # barriers; its wall time is synchronization-inclusive.
            diagnostics["fd_private_restore_synchronization_inclusive_seconds"] = (
                time.perf_counter() - restore_started
            )
            streaming_merger = (
                _StreamingRegisteredLookPacketMerger(self.request)
                if bounded_memory and not aggregate_reduced_score_on_device
                else None
            )
            score_look_accumulator = (
                _DeviceRegisteredLookScoreAccumulator(
                    self.request,
                    device=self.request.packet_plans[
                        0
                    ].reduced_request.declared_device,
                )
                if aggregate_reduced_score_on_device
                else None
            )
            for packet_ordinal, (plan, measurement_request) in enumerate(
                zip(self.request.packet_plans, requests, strict=True)
            ):
                row_object_witness = (
                    _DualChartChunkObjectSeparationWitness()
                    if dual_chart
                    else None
                )
                score = (
                    _LookAggregatingReducedPacketLossOnlyAdapter(
                        plan.reduced_request,
                        packet_ordinal=packet_ordinal,
                        look_accumulator=score_look_accumulator,
                        dual_chunk_observer=(
                            None
                            if row_object_witness is None
                            else row_object_witness.observe
                        ),
                    )
                    if score_look_accumulator is not None
                    else V1SampledReducedPacketLossOnlyAdapter(
                        plan.reduced_request,
                        dual_chunk_observer=(
                            None
                            if row_object_witness is None
                            else row_object_witness.observe
                        ),
                    )
                )
                fd = V1SampledFDLossOnlyAdapter(
                    plan.fd_request,
                    accumulator=private_state,
                    dual_chunk_observer=(
                        None
                        if row_object_witness is None
                        else row_object_witness.observe
                    ),
                )
                row_custody_bridge: _DualRowGroupPacketBridge | None = None
                if row_custody:
                    assert row_group_transaction is not None
                    group_plan = row_group_transaction.plan
                    group_packet_plan = group_plan.packet_plans[packet_ordinal]
                    row_custody_bridge = _DualRowGroupPacketBridge(
                        group_plan_hash=group_plan.group_plan_hash,
                        group_packet_plan=group_packet_plan,
                        staging_consumer=(
                            row_group_transaction.open_packet_staging_consumer(
                                packet_ordinal=packet_ordinal
                            )
                        ),
                        route_ids=plan.reduced_request.route_ids,
                        device=score.device,
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
                    )
                # Child order is frozen: reduced evidence closes before the FD
                # append.  Either output remains private until the whole look
                # and its provider receipts have verified.  Canonical v4/v5 use
                # distinct producer-derived rows from one capture/VJP; legacy
                # requests retain the historical one-row-stream tee.  V5 adds
                # one private custody sink to each nominal branch.
                tee = (
                    build_route_chunked_dual_chart_fanout(
                        evidence_consumers=(
                            (score, row_custody_bridge.evidence_consumer)
                            if row_custody_bridge is not None
                            else (score,)
                        ),
                        jacobian_consumers=(
                            (fd, row_custody_bridge.jacobian_consumer)
                            if row_custody_bridge is not None
                            else (fd,)
                        ),
                    )
                    if dual_chart
                    else _RouteChunkedLossOnlyTee((score, fd))
                )
                provider_started = time.perf_counter()
                provider_receipt = execute(
                    measurement_request,
                    accumulator=tee,
                )
                provider_call_seconds.append(time.perf_counter() - provider_started)
                if not isinstance(provider_receipt, Mapping):
                    raise TypeError("sampled look provider returned no typed receipt")
                row_object_separation_receipt_hash = (
                    None
                    if row_object_witness is None
                    else row_object_witness.close(
                        expected_pair_count=len(score.receipt.route_windows),
                    )
                )
                verification_started = time.perf_counter()
                verified_provider = _verify_provider_receipt(
                    provider_receipt,
                    plan=plan,
                    score=score,
                    fd=fd,
                    row_object_separation_receipt_hash=(
                        row_object_separation_receipt_hash
                    ),
                    row_custody_bridge=row_custody_bridge,
                )
                provider_hashes.append(verified_provider.provider_receipt_hash)
                if dual_chart:
                    if (
                        verified_provider.corrected_evidence_child_receipt_hash
                        is None
                        or verified_provider.literal_jacobian_child_receipt_hash
                        is None
                        or verified_provider.raw_dual_stream_receipt_hash is None
                        or verified_provider.row_object_separation_receipt_hash
                        is None
                    ):
                        raise RuntimeError(
                            "sampled dual-chart provider verification was partial"
                        )
                    evidence_child_receipt_hashes.append(
                        verified_provider.corrected_evidence_child_receipt_hash
                    )
                    jacobian_child_receipt_hashes.append(
                        verified_provider.literal_jacobian_child_receipt_hash
                    )
                    raw_dual_receipt_hashes.append(
                        verified_provider.raw_dual_stream_receipt_hash
                    )
                    row_object_separation_receipt_hashes.append(
                        verified_provider.row_object_separation_receipt_hash
                    )
                    if row_custody:
                        if (
                            verified_provider.corrected_custody_branch_receipt_hash
                            is None
                            or verified_provider.literal_j_custody_branch_receipt_hash
                            is None
                        ):
                            raise RuntimeError(
                                "sampled v5 provider custody verification was partial"
                            )
                        corrected_custody_branch_receipt_hashes.append(
                            verified_provider.corrected_custody_branch_receipt_hash
                        )
                        literal_j_custody_branch_receipt_hashes.append(
                            verified_provider.literal_j_custody_branch_receipt_hash
                        )
                    dual_capture_count += verified_provider.capture_batch_count
                    dual_vjp_count += verified_provider.vjp_count
                diagnostics["provider_receipt_verification_seconds"] = float(
                    diagnostics.get("provider_receipt_verification_seconds", 0.0)
                ) + (time.perf_counter() - verification_started)
                _accumulate_runtime_diagnostics(
                    diagnostics,
                    score.runtime_diagnostics,
                )
                _accumulate_runtime_diagnostics(
                    diagnostics,
                    fd.runtime_diagnostics,
                )
                packet: V1GPUReducedProposalPacket | None = None
                if score_look_accumulator is not None:
                    packet_hashes.append(score.contribution_hash)
                else:
                    packet = score.closed_packet
                if packet is not None and streaming_merger is None:
                    packets.append(packet)
                    packet_hashes.append(packet.packet_hash)
                elif packet is not None:
                    packet_hashes.append(
                        _merge_bounded_registered_look_packet(
                            merger=streaming_merger,
                            plan=plan,
                            packet=packet,
                        )
                    )
                score_closure = score.closure
                append_result = fd.append_result
                fd_closure = fd.closure
                if row_custody_bridge is not None:
                    row_custody_bridge.seal()
                    assert row_group_transaction is not None
                    row_group_transaction.mark_packet_proofs_closed(
                        packet_ordinal=packet_ordinal,
                        provider_proof_hash=(
                            verified_provider.provider_receipt_hash
                        ),
                        reducer_proof_hashes=(
                            score_closure.closure_hash,
                            fd_closure.closure_hash,
                        ),
                    )
                score_closures.append(score_closure)
                append_results.append(append_result)
                fd_closures.append(fd_closure)
                if bounded_memory:
                    # Drop every transaction-owned reference to the closed
                    # route-scale packet before the next provider packet.  The
                    # append-only ledgers above contain hashes/metadata only.
                    if packet is not None:
                        del packet
                    del score
                    del fd
                    del tee
                    del provider_receipt
            snapshot_started = time.perf_counter()
            snapshot = private_state.snapshot()
            # Snapshot materialization reaches existing D2H barriers.  Do not
            # insert any diagnostic-only CUDA synchronization.
            diagnostics["fd_snapshot_synchronization_inclusive_seconds"] = (
                time.perf_counter() - snapshot_started
            )
            _assert_saturated_fd_carry(
                request=self.request,
                snapshot=snapshot,
            )
            # The portable snapshot is now the sole continuation authority.
            # Drop the transaction-private device state before host moment
            # canonicalization and commit construction; retaining its complete
            # route-scale buffers until the return statement needlessly kept
            # roughly 56 GiB allocated at the production rank-32 route axis.
            private_state = None
            snapshot_array_count, snapshot_array_bytes = (
                _fd_snapshot_array_footprint(snapshot)
            )
            diagnostics["fd_snapshot_array_count"] = snapshot_array_count
            diagnostics["fd_snapshot_array_bytes"] = snapshot_array_bytes
            if score_look_accumulator is not None:
                canonicalization_started = time.perf_counter()
                moments = score_look_accumulator.close_moments()
                canonicalization_seconds = time.perf_counter() - canonicalization_started
                merge_seconds = score_look_accumulator.packet_merge_seconds
                packet_input_bytes = score_look_accumulator.packet_input_bytes
                peak_packet_input_bytes = score_look_accumulator.peak_packet_input_bytes
                aggregate_array_bytes = score_look_accumulator.aggregate_array_bytes
                _accumulate_runtime_diagnostics(
                    diagnostics,
                    score_look_accumulator.runtime_diagnostics,
                )
                score_look_accumulator = None
            elif streaming_merger is None:
                moments, merge_seconds, canonicalization_seconds = (
                    _merge_registered_look_packets(
                        request=self.request,
                        packets=packets,
                    )
                )
                packet_input_bytes = sum(
                    int(array.nbytes)
                    for packet in packets
                    for array in (
                        packet.signed_mean_by_stratum,
                        packet.signed_m2_by_stratum,
                        packet.absolute_mean_by_stratum,
                        packet.absolute_m2_by_stratum,
                    )
                )
            else:
                canonicalization_started = time.perf_counter()
                moments = streaming_merger.close()
                canonicalization_seconds = (
                    time.perf_counter() - canonicalization_started
                )
                merge_seconds = streaming_merger.packet_merge_seconds
                packet_input_bytes = streaming_merger.packet_input_bytes
                peak_packet_input_bytes = streaming_merger.peak_packet_input_bytes
                aggregate_array_bytes = streaming_merger.aggregate_array_bytes
                # The immutable moments constructor has copied the aggregate
                # arrays.  Release the mutable merge buffers before commit
                # construction so only the final aggregate remains live.
                streaming_merger = None
            diagnostics["moments_packet_merge_seconds"] = merge_seconds
            diagnostics["moments_canonicalization_seconds"] = (
                canonicalization_seconds
            )
            diagnostics["moments_packet_input_bytes"] = packet_input_bytes
            diagnostics["reduced_packet_array_retention_enabled"] = int(
                retain_reduced_packets
            )
            diagnostics["bounded_host_memory_mode"] = int(bounded_memory)
            if bounded_memory:
                diagnostics["bounded_live_reduced_packet_count_peak"] = (
                    0 if aggregate_reduced_score_on_device else 1
                )
                diagnostics["bounded_retained_reduced_packet_count_at_commit"] = 0
                diagnostics["moments_packet_peak_live_bytes"] = (
                    peak_packet_input_bytes
                )
                diagnostics["moments_merge_aggregate_array_bytes"] = (
                    aggregate_array_bytes
                )
                diagnostics[
                    "bounded_merge_route_scale_array_bytes_upper_bound"
                ] = (
                    aggregate_array_bytes
                    if aggregate_reduced_score_on_device
                    else max(
                        aggregate_array_bytes + peak_packet_input_bytes,
                        2 * aggregate_array_bytes,
                    )
                )
            diagnostics["moments_state_array_bytes"] = sum(
                int(array.nbytes)
                for array in (
                    moments.signed_mean_by_stratum,
                    moments.signed_m2_by_stratum,
                    moments.absolute_mean_by_stratum,
                    moments.absolute_m2_by_stratum,
                )
            )
            commit_started = time.perf_counter()
            dual_receipt = (
                V1SampledDualChartRegisteredLookReceipt(
                    transaction_request_hash=self.request.request_hash,
                    packet_plan_hashes=tuple(
                        plan.plan_hash for plan in self.request.packet_plans
                    ),
                    provider_receipt_hashes=tuple(provider_hashes),
                    corrected_evidence_child_receipt_hashes=tuple(
                        evidence_child_receipt_hashes
                    ),
                    literal_jacobian_child_receipt_hashes=tuple(
                        jacobian_child_receipt_hashes
                    ),
                    raw_dual_stream_receipt_hashes=tuple(raw_dual_receipt_hashes),
                    row_object_separation_receipt_hashes=tuple(
                        row_object_separation_receipt_hashes
                    ),
                    corrected_evidence_stream_identity_hash=(
                        self.request.packet_plans[
                            0
                        ].corrected_evidence_stream_identity_hash
                    ),
                    literal_jacobian_stream_identity_hash=(
                        self.request.packet_plans[0].literal_jacobian_stream_identity_hash
                    ),
                    packet_count=len(self.request.packet_plans),
                    capture_batch_count=dual_capture_count,
                    vjp_count=dual_vjp_count,
                )
                if dual_chart
                else None
            )
            prepared_row_group = (
                row_group_transaction.prepare_commit()
                if row_group_transaction is not None
                else None
            )
            row_custody_receipt = (
                V1SampledDualChartRowCustodyRegisteredLookReceipt(
                    transaction_request_hash=self.request.request_hash,
                    registered_packet_plan_hashes=tuple(
                        plan.plan_hash for plan in self.request.packet_plans
                    ),
                    dual_chart_receipt_hash=dual_receipt.receipt_hash,
                    group_plan_hash=(
                        self.request.dual_chart_row_group_plan.group_plan_hash
                    ),
                    group_commit_receipt=prepared_row_group.receipt,
                    provider_receipt_hashes=tuple(provider_hashes),
                    corrected_reducer_closure_hashes=tuple(
                        value.closure_hash for value in score_closures
                    ),
                    literal_j_reducer_closure_hashes=tuple(
                        value.closure_hash for value in fd_closures
                    ),
                    corrected_custody_branch_receipt_hashes=tuple(
                        corrected_custody_branch_receipt_hashes
                    ),
                    literal_j_custody_branch_receipt_hashes=tuple(
                        literal_j_custody_branch_receipt_hashes
                    ),
                    corrected_evidence_stream_identity_hash=(
                        self.request.packet_plans[
                            0
                        ].corrected_evidence_stream_identity_hash
                    ),
                    literal_jacobian_stream_identity_hash=(
                        self.request.packet_plans[
                            0
                        ].literal_jacobian_stream_identity_hash
                    ),
                    packet_count=len(self.request.packet_plans),
                )
                if row_custody
                else None
            )
            result = V1SampledRegisteredLookCommit(
                request=self.request,
                reduced_packets=tuple(packets) if retain_reduced_packets else (),
                reduced_packet_hashes=tuple(packet_hashes),
                reduced_closures=tuple(score_closures),
                fd_append_results=tuple(append_results),
                fd_closures=tuple(fd_closures),
                provider_receipt_hashes=tuple(provider_hashes),
                corrected_attribution_moments=moments,
                fd_snapshot=snapshot,
                dual_chart_receipt=dual_receipt,
                dual_chart_row_custody_receipt=row_custody_receipt,
                canonical_q_input_authority=dual_chart,
                schema=(
                    _commit_schema_for_request_schema(self.request.schema)
                ),
                _trust_cached_reduced_packet_hashes=True,
            )
            diagnostics["commit_construction_seconds"] = (
                time.perf_counter() - commit_started
            )
            integrity_started = time.perf_counter()
            result.commit_hash
            diagnostics["commit_integrity_seconds"] = (
                time.perf_counter() - integrity_started
            )
            diagnostics[
                "provider_vjp_call_synchronization_inclusive_seconds_by_packet"
            ] = tuple(provider_call_seconds)
            diagnostics[
                "provider_vjp_call_synchronization_inclusive_seconds"
            ] = sum(provider_call_seconds)
            diagnostics["provider_vjp_call_count"] = len(provider_call_seconds)
            diagnostics["registered_transport_packet_count"] = len(
                self.request.packet_plans
            )
            diagnostics["transaction_seconds"] = (
                time.perf_counter() - transaction_started
            )
            for name, raw_value in diagnostics.items():
                values = raw_value if isinstance(raw_value, tuple) else (raw_value,)
                for value in values:
                    if isinstance(value, bool) or not isinstance(
                        value, (int, float, np.integer, np.floating)
                    ):
                        raise TypeError(
                            f"sampled runtime diagnostic {name} is not numeric"
                        )
                    if not math.isfinite(float(value)) or float(value) < 0.0:
                        raise ValueError(
                            f"sampled runtime diagnostic {name} is invalid"
                        )
            if row_group_transaction is not None:
                assert prepared_row_group is not None
                prepared_outer = V1SampledRegisteredLookPreparedCommitV5(
                    commit=result,
                    runtime_diagnostics=diagnostics,
                    _group_transaction=row_group_transaction,
                    _owner=self,
                    _issuance_token=_PREPARED_V5_ISSUANCE_TOKEN,
                )
                self._prepared_v5 = prepared_outer
                private_state = None
                return prepared_outer
            self._result = result
            self._runtime_diagnostics = MappingProxyType(dict(diagnostics))
            self._dual_chart_row_prefix_attachment = None
            self._closed = True
            private_state = None
            return result
        except Exception as error:
            # The only continuation state visible to the caller is still the
            # request's immutable prior snapshot.  Do not retain partial child
            # products even as convenient diagnostics on this authority type.
            packets.clear()
            packet_hashes.clear()
            score_closures.clear()
            append_results.clear()
            fd_closures.clear()
            provider_hashes.clear()
            evidence_child_receipt_hashes.clear()
            jacobian_child_receipt_hashes.clear()
            raw_dual_receipt_hashes.clear()
            corrected_custody_branch_receipt_hashes.clear()
            literal_j_custody_branch_receipt_hashes.clear()
            if row_group_transaction is not None:
                row_group_transaction.abort()
            streaming_merger = None
            score_look_accumulator = None
            private_state = None
            self._runtime_diagnostics = None
            self._dual_chart_row_prefix_attachment = None
            self._failed = True
            self._failure_reason = f"{type(error).__name__}: {error}"
            raise


__all__ = [
    "SAMPLED_DEVICE_RESIDENT_D_TRANSACTION_RESULT_SCHEMA_V1",
    "SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA",
    "SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V1",
    "SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V3",
    "SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V4",
    "SAMPLED_REGISTERED_LOOK_COMMIT_SCHEMA_V5",
    "SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA",
    "SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V1",
    "SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V3",
    "SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4",
    "SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V5",
    "SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V6",
    "SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND",
    "SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V1",
    "SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V3",
    "SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V4",
    "SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V5",
    "SAMPLED_REGISTERED_LOOK_TRANSACTION_KIND_V6",
    "SAMPLED_LOOK_PACKET_PLAN_SCHEMA_V2",
    "V1SampledDeviceResidentDTransactionResult",
    "V1SampledDualChartRegisteredLookReceipt",
    "V1SampledDualChartChunkObjectSeparationWitness",
    "V1SampledDualChartRowCustodyRegisteredLookReceipt",
    "V1SampledLookPacketPlan",
    "V1SampledRegisteredLookCommit",
    "V1SampledRegisteredLookPreparedCommitV5",
    "V1SampledRegisteredLookTransaction",
    "V1SampledRegisteredLookTransactionRequest",
    "V1SampledDualRowGroupPacketBridge",
    "verify_v1_sampled_dual_chart_custody_only_provider_receipt",
]
