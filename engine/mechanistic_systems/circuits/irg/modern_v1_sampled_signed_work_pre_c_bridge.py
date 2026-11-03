"""Bind restored sampled corrected-A custody to the canonical pre-C source.

The physical registered-D request, sampled KKT observation, and restored
signed-work row bank intentionally name distinct identities.  This seam is the
only place where an exact corrected-A operator may adopt the prepared
problem's ``source_geometry_hash``.  It does so only after replaying an exact
physical-request-to-observation crosswalk and the original packet ledgers.

The returned source is proposal-input-only.  It observes no C, K, O, or
validation outcome and grants no certification, commit, terminal, or Pareto
authority.  The live row bank is consumed by the result: any failed bind
evicts it, and successful callers release both operator lease and bank through
``V1SampledSignedWorkPreCRegistrationResult.evict``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Final, NoReturn

from .contracts import content_hash
from .modern_v1_relaxed_kkt import V1RelaxedKKTProblem
from .modern_v1_sampled_composition_backend import (
    V1SampledCompositionRegisteredDTransaction,
)
from .modern_v1_sampled_kkt_bridge import V1SampledKKTSourceProblemBundle
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJOriginalCaptureBinding,
)
from .modern_v1_sampled_signed_work_row_bank_cuda import (
    V1SampledSignedWorkRowBankRestorationResult,
)
from .modern_v1_signed_work_operator_cuda import (
    V1SignedWorkOperatorCUDA,
    V1SignedWorkOperatorCUDAExecutionDesign,
)
from .modern_v1_signed_work_pre_c_coordinator import (
    V1SignedWorkPreCSourceBundle,
    _issue_v1_signed_work_pre_c_source_bundle_from_sampled_registration,
    _issue_v1_signed_work_pre_c_source_bundle_private_cpu_reference_for_tests,
)


SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_signed_work_pre_C_registration_v2"
)
SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_AUTHORITY_SCOPE: Final = (
    "same_D_corrected_A_signed_work_operator_to_pre_C_proposal_source_only_"
    "no_FQR_certification_C_K_O_calibration_oracle_commit_terminal_or_"
    "pareto_authority_v1"
)

_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-pre-C-registration-receipt-"
)
_SAMPLED_OBSERVATION_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-observation-prefix-"
)
_ALEXNET_REGISTERED_D_RAW_PREFIX_SCHEMA: Final = (
    "partition_free_v1_alexnet_registered_D_raw_prefix_coordinate_v1"
)
_ALEXNET_REGISTERED_D_SLICE_PREFIX: Final = (
    "partition-free-v1-alexnet-registered-D-context-draw-slice-"
)
_ALEXNET_REGISTERED_D_SLICE_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-registered-D-slice-axis-"
)
_ALEXNET_REGISTERED_D_CROSSWALK_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-registered-D-crosswalk-"
)
_ALEXNET_LITERAL_J_PREFIX_SOURCE_SCHEMA: Final = (
    "partition_free_v1_alexnet_registered_D_literal_J_prefix_source_receipt_v1"
)
_ALEXNET_LITERAL_J_PREFIX_SOURCE_PREFIX: Final = (
    "partition-free-v1-alexnet-registered-D-literal-J-prefix-source-"
)
_ROW_BANK_SOURCE_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-signed-work-original-source-measurement-axis-"
)
_SAMPLED_DRAW_AXIS_PREFIX: Final = "partition-free-v1-sampled-draw-axis-"
_SAMPLED_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-sampled-source-context-axis-"
_SAMPLED_MEASUREMENT_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-source-measurement-axis-"
)
_RELAXED_PROBLEM_PREFIX: Final = "partition-free-v1-relaxed-kkt-problem-"
_ISSUANCE_TOKEN: Final = object()
_RESULT_TOKEN: Final = object()


class V1SampledSignedWorkPreCBridgeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PREPARED_BUNDLE_MUTATED = "prepared_bundle_mutated"
    ORIGINAL_CAPTURE_MISMATCH = "original_capture_mismatch"
    STALE_REGISTERED_D_PREFIX = "stale_registered_D_prefix"
    STALE_RAW_PREFIX = "stale_raw_prefix"
    REGISTERED_D_TRANSACTION_MISMATCH = "registered_D_transaction_mismatch"
    REGISTERED_D_CROSSWALK_MISMATCH = "registered_D_crosswalk_mismatch"
    FD_RANK_OR_DESIGN_MISMATCH = "FD_rank_or_design_mismatch"
    CHART_OR_MASK_MISMATCH = "chart_or_mask_mismatch"
    ROUTE_AXIS_MISMATCH = "route_axis_mismatch"
    STREAM_ROLE_MISMATCH = "stream_role_mismatch"
    SOURCE_AXIS_MISMATCH = "source_axis_mismatch"
    SOURCE_GEOMETRY_MISMATCH = "source_geometry_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    OPERATOR_CONSTRUCTION_FAILED = "operator_construction_failed"
    CLEANUP_FAILED = "cleanup_failed"


class V1SampledSignedWorkPreCBridgeRefusal(RuntimeError):
    """Typed fail-closed refusal for the sampled signed-work pre-C seam."""

    def __init__(
        self,
        code: V1SampledSignedWorkPreCBridgeRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledSignedWorkPreCBridgeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledSignedWorkPreCBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _flatten(values: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    return tuple(item for packet in values for item in packet)


def _problem_hash(problem: V1RelaxedKKTProblem) -> str:
    """Recompute the immutable problem hash rather than trusting its cache."""

    return content_hash(
        {
            "schema": problem.schema,
            "problem_id": problem.problem_id,
            "resource": problem.resource,
            "corrected_work": problem.corrected_work,
            "low_rank_factor": problem.low_rank_factor,
            "residual_diagonal": problem.residual_diagonal,
            "residual_spectral_bound": problem.residual_spectral_bound,
            "eligible_mask": problem.eligible_mask,
            "work_budget": problem.work_budget,
            "quadratic_budget": problem.quadratic_budget,
            "route_axis_hash": problem.route_axis_hash,
            "source_geometry_hash": problem.source_geometry_hash,
            "block_capacities": problem.block_capacities,
            "input_array_policy": problem.input_array_policy,
            "corrected_work_envelope_id": problem.corrected_work_envelope_id,
        },
        prefix=_RELAXED_PROBLEM_PREFIX,
    )


def _sampled_observation_prefix_coordinate(
    prepared: V1SampledKKTSourceProblemBundle,
) -> tuple[object, ...]:
    admission = prepared.sampled_geometry_source.admission
    receipt = admission.receipt
    dual = admission.dual_chart_stream_binding
    if dual is None:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.STREAM_ROLE_MISMATCH,
            "canonical signed work requires a sampled corrected-A/literal-J binding",
        )
    corrected_stream, literal_j_stream, _proof_hash = dual
    return (
        receipt.model_identity_hash,
        receipt.current_mask_hash,
        receipt.physical_route_axis_hash,
        receipt.ordered_route_axis_hash,
        receipt.route_count,
        receipt.proposal_generation,
        receipt.observed_stratum_counts,
        receipt.observed_draw_axis_hash,
        receipt.observed_source_context_axis_hash,
        receipt.source_measurement_axis_hash,
        receipt.sample_weight_hash,
        receipt.corrected_moments_state_hash,
        corrected_stream,
        literal_j_stream,
    )


def derive_v1_sampled_signed_work_sampled_observation_prefix_hash(
    prepared: V1SampledKKTSourceProblemBundle,
) -> str:
    """Name the prepared sampled observation, not its physical request prefix.

    The physical registered-D prefix belongs to
    :class:`V1SampledCompositionRegisteredDTransaction`.  This independent
    digest names the admitted moments/snapshot observation.  Keeping the two
    identities distinct prevents a prepared aggregate from impersonating the
    physical request ledger whose raw corrected-A rows are restored.
    """

    if type(prepared) is not V1SampledKKTSourceProblemBundle:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.INVALID_TYPE,
            "observation-prefix derivation requires the exact sampled KKT source problem",
        )
    prepared.assert_integrity()
    return content_hash(
        _sampled_observation_prefix_coordinate(prepared),
        prefix=_SAMPLED_OBSERVATION_PREFIX,
    )


_SLICE_PAYLOAD_KEYS: Final = (
    "registered_look_index",
    "registered_look_cumulative_stratum_counts",
    "sealed_order_start_indices",
    "look_context_ids_by_stratum",
    "look_draw_ids_by_stratum",
    "look_stratum_labels_by_stratum",
    "transport_packet_spans",
    "packet_index",
    "transport_packet_index_after",
    "train_context_ids",
    "train_draw_ids",
)
_SLICE_KEYS: Final = frozenset((*_SLICE_PAYLOAD_KEYS, "slice_hash"))


def _crosswalk_refuse(message: str) -> NoReturn:
    _refuse(
        V1SampledSignedWorkPreCBridgeRefusalCode.REGISTERED_D_CROSSWALK_MISMATCH,
        message,
    )


def _exact_pair_of_counts(value: object, *, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        _crosswalk_refuse(f"{name} must be one nonnegative integer pair")
    return value  # type: ignore[return-value]


def _exact_nested_string_pair(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(value) is not tuple or len(value) != 2:
        _crosswalk_refuse(f"{name} must be one two-stratum tuple")
    rows = value
    if any(
        type(row) is not tuple or any(type(item) is not str or not item for item in row)
        for row in rows
    ):
        _crosswalk_refuse(f"{name} contains a noncanonical string row")
    return rows  # type: ignore[return-value]


def _exact_string_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item for item in value
    ):
        _crosswalk_refuse(f"{name} must be one exact string tuple")
    return value  # type: ignore[return-value]


def _validate_registered_d_transaction_crosswalk(
    *,
    prepared: V1SampledKKTSourceProblemBundle,
    transaction: V1SampledCompositionRegisteredDTransaction,
) -> dict[str, object]:
    """Close the physical AlexNet request ledger against sampled moments.

    The transaction's hash names the physical request prefix.  The prepared
    source names aggregate sampled observations.  Equality between those two
    hashes is neither expected nor accepted as a substitute for this explicit
    coordinate/slice crosswalk.
    """

    try:
        transaction.design.__post_init__()
        transaction.__post_init__()
        transaction_hash = transaction.transaction_hash
    except BaseException as error:
        raise V1SampledSignedWorkPreCBridgeRefusal(
            V1SampledSignedWorkPreCBridgeRefusalCode.REGISTERED_D_TRANSACTION_MISMATCH,
            "physical registered-D transaction failed exact integrity replay",
        ) from error

    admission = prepared.sampled_geometry_source.admission
    moments = admission.moments
    source_receipt = admission.receipt
    fd_design = admission.fd_design
    fd_snapshot = admission.fd_snapshot
    selected_rank = transaction.physical_fd_rank
    if (
        transaction.fd_design_hash != fd_design.design_hash
        or source_receipt.fd_design_hash != fd_design.design_hash
        or fd_design.spec.initial_rank != selected_rank
        or fd_design.spec.maximum_rank != selected_rank
        or fd_snapshot.sketch_rank != selected_rank
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.FD_RANK_OR_DESIGN_MISMATCH,
            "physical transaction rank/design differs from the prepared FD observation",
        )

    coordinate = transaction.rank_invariant_raw_prefix_coordinate
    if type(coordinate) is not tuple or len(coordinate) != 13:
        _crosswalk_refuse(
            "physical registered-D coordinate is not the exact 13-field AlexNet ABI"
        )
    (
        schema,
        campaign_hash,
        model_hash,
        query_hash,
        preprocessing_hash,
        mask_hash,
        ordered_route_axis_hash,
        sampling_law_hash,
        proposal_generation,
        acquisition_design_hash,
        raw_look_rows,
        corrected_stream,
        literal_j_stream,
    ) = coordinate
    expected_coordinate_head = (
        _ALEXNET_REGISTERED_D_RAW_PREFIX_SCHEMA,
        moments.campaign_contract_hash,
        moments.model_identity_hash,
        moments.query_contract_hash,
        moments.preprocessing_hash,
        moments.current_mask_hash,
        source_receipt.ordered_route_axis_hash,
        moments.sampling_law_hash,
        moments.proposal_generation,
    )
    if (
        (
            schema,
            campaign_hash,
            model_hash,
            query_hash,
            preprocessing_hash,
            mask_hash,
            ordered_route_axis_hash,
            sampling_law_hash,
            proposal_generation,
        )
        != expected_coordinate_head
        or source_receipt.ordered_route_axis_hash != moments.route_axis_hash
    ):
        _crosswalk_refuse(
            "physical registered-D chart coordinate differs from prepared sampled D"
        )
    acquisition_design = _identity(
        acquisition_design_hash,
        name="registered sequential acquisition design hash",
    )
    source_origin_request = getattr(admission.source_origin, "request", None)
    if source_origin_request is not None and (
        getattr(source_origin_request, "registered_sequential_design_hash", None)
        != acquisition_design
    ):
        _crosswalk_refuse(
            "physical acquisition design differs from the live sampled-look source"
        )
    dual = admission.dual_chart_stream_binding
    if dual is None:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.STREAM_ROLE_MISMATCH,
            "prepared sampled D lacks its dual corrected-A/literal-J streams",
        )
    expected_corrected_stream, expected_literal_j_stream, _dual_proof = dual
    if (
        corrected_stream != expected_corrected_stream
        or literal_j_stream != expected_literal_j_stream
        or corrected_stream != moments.raw_logit_stream_source_hash
        or literal_j_stream != fd_design.raw_logit_stream_source_hash
        or corrected_stream == literal_j_stream
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.STREAM_ROLE_MISMATCH,
            "physical registered-D coordinate changed the dual stream roles",
        )

    if (
        type(raw_look_rows) is not tuple
        or len(raw_look_rows) != transaction.registered_look_count
        or transaction.registered_look_count != moments.look_index
        or transaction.replay_look_indices != tuple(range(1, moments.look_index + 1))
    ):
        _crosswalk_refuse(
            "physical registered-D look ledger is not the complete contiguous sampled prefix"
        )

    names = moments.stratum_names
    reconstructed_contexts: list[list[str]] = [[], []]
    reconstructed_draws: list[list[str]] = [[], []]
    prior_counts = (0, 0)
    prior_transport_packet_index = 0
    slice_hashes: list[str] = []
    for expected_look, raw_row in enumerate(raw_look_rows, start=1):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _SLICE_KEYS:
            _crosswalk_refuse(
                "registered-D look row changed the exact AlexNet slice schema"
            )
        slice_payload = {key: raw_row[key] for key in _SLICE_PAYLOAD_KEYS}
        observed_slice_hash = _identity(
            raw_row["slice_hash"],
            name="registered-D context/draw slice hash",
        )
        expected_slice_hash = content_hash(
            slice_payload,
            prefix=_ALEXNET_REGISTERED_D_SLICE_PREFIX,
        )
        if observed_slice_hash != expected_slice_hash:
            _crosswalk_refuse(
                "registered-D look row differs from its physical slice hash"
            )
        if observed_slice_hash in slice_hashes:
            _crosswalk_refuse("registered-D slice hash ledger contains a duplicate")
        slice_hashes.append(observed_slice_hash)

        look_index = raw_row["registered_look_index"]
        if type(look_index) is not int or look_index != expected_look:
            _crosswalk_refuse(
                "registered-D look indices are not the contiguous prefix from one"
            )
        starts = _exact_pair_of_counts(
            raw_row["sealed_order_start_indices"],
            name="sealed-order start indices",
        )
        stops = _exact_pair_of_counts(
            raw_row["registered_look_cumulative_stratum_counts"],
            name="registered-look cumulative counts",
        )
        if (
            starts != prior_counts
            or any(stop < start for start, stop in zip(starts, stops, strict=True))
            or not any(stop > start for start, stop in zip(starts, stops, strict=True))
        ):
            _crosswalk_refuse(
                "registered-D look counts contain a gap, rollback, or empty look"
            )
        look_contexts = _exact_nested_string_pair(
            raw_row["look_context_ids_by_stratum"],
            name="look context IDs by stratum",
        )
        look_draws = _exact_nested_string_pair(
            raw_row["look_draw_ids_by_stratum"],
            name="look draw IDs by stratum",
        )
        look_labels = _exact_nested_string_pair(
            raw_row["look_stratum_labels_by_stratum"],
            name="look stratum labels by stratum",
        )
        increments = tuple(
            stop - start for start, stop in zip(starts, stops, strict=True)
        )
        if any(
            len(look_contexts[index]) != increments[index]
            or len(look_draws[index]) != increments[index]
            or look_labels[index] != (names[index],) * increments[index]
            for index in (0, 1)
        ):
            _crosswalk_refuse(
                "registered-D context/draw/stratum slices do not realize their counts"
            )
        train_contexts = _exact_string_axis(
            raw_row["train_context_ids"],
            name="registered-D train context axis",
        )
        train_draws = _exact_string_axis(
            raw_row["train_draw_ids"],
            name="registered-D train draw axis",
        )
        if train_contexts != _flatten(look_contexts) or train_draws != _flatten(
            look_draws
        ):
            _crosswalk_refuse(
                "registered-D flat train axes differ from their stratum slices"
            )

        spans = raw_row["transport_packet_spans"]
        packet_index = raw_row["packet_index"]
        packet_index_after = raw_row["transport_packet_index_after"]
        if (
            type(spans) is not tuple
            or not spans
            or type(packet_index) is not int
            or type(packet_index_after) is not int
            or packet_index != prior_transport_packet_index
            or packet_index_after != packet_index + len(spans)
        ):
            _crosswalk_refuse(
                "registered-D transport packet boundary is not contiguous"
            )
        span_cursors = list(starts)
        prior_stratum_index = -1
        for span in spans:
            if (
                type(span) is not tuple
                or len(span) != 3
                or span[0] not in names
                or type(span[1]) is not int
                or type(span[2]) is not int
            ):
                _crosswalk_refuse("registered-D transport span is malformed")
            stratum_index = names.index(span[0])
            left, right = span[1], span[2]
            if (
                stratum_index < prior_stratum_index
                or left != span_cursors[stratum_index]
                or right <= left
                or right > stops[stratum_index]
            ):
                _crosswalk_refuse(
                    "registered-D transport spans are not grouped contiguous covers"
                )
            span_cursors[stratum_index] = right
            prior_stratum_index = stratum_index
        if tuple(span_cursors) != stops:
            _crosswalk_refuse(
                "registered-D transport spans do not cover the complete look"
            )

        for index in (0, 1):
            reconstructed_contexts[index].extend(look_contexts[index])
            reconstructed_draws[index].extend(look_draws[index])
        prior_counts = stops
        prior_transport_packet_index = packet_index_after

    contexts_by_stratum = tuple(tuple(row) for row in reconstructed_contexts)
    draws_by_stratum = tuple(tuple(row) for row in reconstructed_draws)
    if (
        prior_counts != moments.stratum_counts
        or contexts_by_stratum != moments.observed_context_ids_by_stratum
        or draws_by_stratum != moments.observed_draw_ids_by_stratum
    ):
        _crosswalk_refuse(
            "physical registered-D slices reconstruct another sampled observation"
        )
    if transaction.transaction_hash != transaction_hash:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.REGISTERED_D_TRANSACTION_MISMATCH,
            "physical registered-D transaction changed during crosswalk replay",
        )

    expected_literal_j_source_receipt_hash = content_hash(
        {
            "schema": _ALEXNET_LITERAL_J_PREFIX_SOURCE_SCHEMA,
            "campaign_contract_hash": campaign_hash,
            "model_identity_hash": model_hash,
            "query_contract_hash": query_hash,
            "preprocessing_hash": preprocessing_hash,
            "current_mask_hash": mask_hash,
            "route_axis_hash": ordered_route_axis_hash,
            "sampling_law_hash": sampling_law_hash,
            "proposal_generation": proposal_generation,
            "registered_sequential_design_hash": acquisition_design,
            "corrected_evidence_stream_identity_hash": corrected_stream,
            "literal_jacobian_stream_identity_hash": literal_j_stream,
            "registered_look_context_draw_slice_hashes": tuple(slice_hashes),
            "rank_invariant_raw_prefix_hash": (
                transaction.rank_invariant_raw_prefix_hash
            ),
            "same_capture_and_vjp_required": True,
            "source_provenance_authority": True,
            "exact_operator_geometry_uncertainty_authority": False,
            "fixed_direction_geometry_uncertainty_authority": False,
            "C_authority": False,
            "calibration_authority": False,
            "oracle_authority": False,
            "mask_commit_authority": False,
        },
        prefix=_ALEXNET_LITERAL_J_PREFIX_SOURCE_PREFIX,
    )
    if (
        transaction.literal_j_source_receipt_hash
        != expected_literal_j_source_receipt_hash
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.REGISTERED_D_TRANSACTION_MISMATCH,
            "physical transaction literal-J source receipt does not replay from its coordinate",
        )

    sampled_observation_prefix_hash = content_hash(
        _sampled_observation_prefix_coordinate(prepared),
        prefix=_SAMPLED_OBSERVATION_PREFIX,
    )
    slice_axis_hash = content_hash(
        tuple(slice_hashes),
        prefix=_ALEXNET_REGISTERED_D_SLICE_AXIS_PREFIX,
    )
    crosswalk_hash = content_hash(
        {
            "physical_d_transaction_hash": transaction_hash,
            "physical_rank_invariant_raw_prefix_hash": (
                transaction.rank_invariant_raw_prefix_hash
            ),
            "physical_literal_j_source_receipt_hash": (
                expected_literal_j_source_receipt_hash
            ),
            "sampled_observation_prefix_hash": sampled_observation_prefix_hash,
            "selected_physical_fd_rank": selected_rank,
            "selected_fd_design_hash": transaction.fd_design_hash,
            "registered_sequential_acquisition_design_hash": acquisition_design,
            "registered_look_slice_hashes": tuple(slice_hashes),
            "registered_look_slice_axis_hash": slice_axis_hash,
            "observed_context_ids_by_stratum": contexts_by_stratum,
            "observed_draw_ids_by_stratum": draws_by_stratum,
        },
        prefix=_ALEXNET_REGISTERED_D_CROSSWALK_PREFIX,
    )
    return {
        "transaction_hash": transaction_hash,
        "physical_raw_prefix_hash": transaction.rank_invariant_raw_prefix_hash,
        "literal_j_source_receipt_hash": expected_literal_j_source_receipt_hash,
        "sampled_observation_prefix_hash": sampled_observation_prefix_hash,
        "selected_rank": selected_rank,
        "selected_fd_design_hash": transaction.fd_design_hash,
        "acquisition_design_hash": acquisition_design,
        "slice_hashes": tuple(slice_hashes),
        "slice_axis_hash": slice_axis_hash,
        "crosswalk_hash": crosswalk_hash,
    }


def _group_original_axis(
    *,
    values: tuple[tuple[str, ...], ...],
    labels: tuple[tuple[str, ...], ...],
    stratum_names: tuple[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    flat_values = _flatten(values)
    flat_labels = _flatten(labels)
    if len(flat_values) != len(flat_labels):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.SOURCE_AXIS_MISMATCH,
            "original packet value and stratum axes have different lengths",
        )
    return tuple(
        tuple(
            value
            for value, label in zip(flat_values, flat_labels, strict=True)
            if label == stratum
        )
        for stratum in stratum_names
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class V1SampledSignedWorkPreCJointRegistrationReceipt:
    """Path-free proof authorizing one exact same-D pre-C source registration."""

    prepared_source_problem_hash: str
    sampled_geometry_source_hash: str
    source_admission_hash: str
    response_geometry_problem_hash: str
    response_geometry_source_hash: str
    original_capture_binding_hash: str
    restoration_receipt_hash: str
    row_spool_custody_receipt_hash: str
    row_prefix_manifest_hash: str
    registered_d_prefix_hash: str
    registered_d_coordinate_hash: str
    physical_d_transaction_hash: str
    physical_rank_invariant_raw_prefix_hash: str
    physical_literal_j_source_receipt_hash: str
    sampled_observation_prefix_hash: str
    selected_physical_fd_rank: int
    selected_fd_design_hash: str
    registered_sequential_acquisition_design_hash: str
    registered_look_slice_hashes: tuple[str, ...]
    registered_look_slice_axis_hash: str
    physical_to_sampled_observation_crosswalk_hash: str
    original_capture_chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    context_count: int
    stratum_names: tuple[str, str]
    observed_stratum_counts: tuple[int, int]
    reconstructed_draw_axis_hash: str
    reconstructed_source_context_axis_hash: str
    reconstructed_source_measurement_axis_hash: str
    original_packet_stratum_axis_hash: str
    original_packet_source_measurement_axis_hash: str
    dual_chart_receipt_hash: str
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    signed_work_operator_receipt_hash: str
    signed_work_operator_design_hash: str
    source_geometry_override_proven_by_same_d_lineage: bool = True
    source_registration_authority: bool = True
    production_cuda_proposal_input_authority: bool = True
    private_cpu_reference_test_only: bool = False
    fqr_certification_authority: bool = False
    c_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_AUTHORITY_SCOPE
    schema: str = SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUANCE_TOKEN:
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "joint registration receipts are issued only by the closed binder",
            )
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            if item.name.endswith("_hash"):
                _identity(getattr(self, item.name), name=item.name.replace("_", " "))
        if (
            type(self.proposal_generation) is not int
            or self.proposal_generation < 0
            or type(self.route_count) is not int
            or self.route_count <= 0
            or type(self.context_count) is not int
            or self.context_count <= 0
            or type(self.selected_physical_fd_rank) is not int
            or self.selected_physical_fd_rank <= 0
            or len(self.stratum_names) != 2
            or len(set(self.stratum_names)) != 2
            or len(self.observed_stratum_counts) != 2
            or any(
                type(value) is not int or value <= 0
                for value in self.observed_stratum_counts
            )
            or sum(self.observed_stratum_counts) != self.context_count
            or not self.registered_look_slice_hashes
            or len(set(self.registered_look_slice_hashes))
            != len(self.registered_look_slice_hashes)
            or any(
                type(value) is not str or not value
                for value in self.registered_look_slice_hashes
            )
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.SOURCE_AXIS_MISMATCH,
                "joint registration dimensions or two-stratum prefix are invalid",
            )
        required_true = (
            self.source_geometry_override_proven_by_same_d_lineage,
            self.source_registration_authority,
        )
        production_pair = (
            self.production_cuda_proposal_input_authority,
            self.private_cpu_reference_test_only,
        )
        downstream = (
            self.fqr_certification_authority,
            self.c_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(
            type(value) is not bool
            for value in (*required_true, *production_pair, *downstream)
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "joint registration authority flags must be exact booleans",
            )
        if (
            not all(required_true)
            or production_pair not in {(True, False), (False, True)}
            or any(downstream)
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or self.physical_rank_invariant_raw_prefix_hash
            == self.sampled_observation_prefix_hash
            or self.authority_scope
            != SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_AUTHORITY_SCOPE
            or self.schema != SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "joint registration changed its proposal-only authority boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: (
                getattr(self, item.name).value
                if isinstance(getattr(self, item.name), Enum)
                else getattr(self, item.name)
            )
            for item in fields(self)
            if not item.name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.ORIGINAL_CAPTURE_MISMATCH,
                "joint registration receipt was mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.ORIGINAL_CAPTURE_MISMATCH,
                "joint registration receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def _evict_live_sources(
    restoration: V1SampledSignedWorkRowBankRestorationResult,
    operator: V1SignedWorkOperatorCUDA | None,
) -> None:
    failures: list[BaseException] = []
    if operator is not None:
        try:
            operator.evict()
        except BaseException as error:  # pragma: no cover - injected failure path.
            failures.append(error)
    handle = restoration.row_bank_handle
    try:
        if not handle.is_evicted:
            handle.evict()
    except BaseException as error:  # pragma: no cover - injected failure path.
        failures.append(error)
    if failures:
        raise V1SampledSignedWorkPreCBridgeRefusal(
            V1SampledSignedWorkPreCBridgeRefusalCode.CLEANUP_FAILED,
            "failed pre-C registration could not release all live custody",
        ) from failures[0]


@dataclass(frozen=True, slots=True)
class V1SampledSignedWorkPreCRegistrationResult:
    """Live pre-C source plus the consumed row-bank custody owner."""

    registration_receipt: V1SampledSignedWorkPreCJointRegistrationReceipt
    source_bundle: V1SignedWorkPreCSourceBundle = field(repr=False, compare=False)
    restoration_result: V1SampledSignedWorkRowBankRestorationResult = field(
        repr=False,
        compare=False,
    )
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _evicted: bool = field(init=False, default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RESULT_TOKEN:
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "live pre-C registration results are binder-issued only",
            )
        if (
            type(self.registration_receipt)
            is not V1SampledSignedWorkPreCJointRegistrationReceipt
            or type(self.source_bundle) is not V1SignedWorkPreCSourceBundle
            or type(self.restoration_result)
            is not V1SampledSignedWorkRowBankRestorationResult
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.INVALID_TYPE,
                "live pre-C registration contains a noncanonical component",
            )
        self.assert_integrity()

    @property
    def signed_work_operator(self) -> V1SignedWorkOperatorCUDA:
        return self.source_bundle.signed_work_operator

    @property
    def source_registration_hash(self) -> str:
        return self.registration_receipt.receipt_hash

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_integrity(self) -> str:
        if self._evicted:
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "evicted pre-C registration cannot be reused",
            )
        receipt_hash = self.registration_receipt.assert_integrity()
        source = self.source_bundle
        restoration_receipt = self.restoration_result.receipt
        if (
            source.source_registration_hash != receipt_hash
            or source.registered_d_coordinate_hash
            != self.registration_receipt.registered_d_coordinate_hash
            or source.signed_work_operator is not self.signed_work_operator
            or source.signed_work_operator.receipt_hash
            != self.registration_receipt.signed_work_operator_receipt_hash
            or restoration_receipt.receipt_hash
            != self.registration_receipt.restoration_receipt_hash
            or restoration_receipt.rank_invariant_raw_prefix_hash
            != self.registration_receipt.physical_rank_invariant_raw_prefix_hash
            or self.restoration_result.row_bank_handle.active_lease_count != 1
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.SOURCE_GEOMETRY_MISMATCH,
                "live operator, receipt, and pre-C source no longer close",
            )
        source.bundle_hash
        return receipt_hash

    def evict(self) -> None:
        if self._evicted:
            return
        _evict_live_sources(self.restoration_result, self.signed_work_operator)
        object.__setattr__(self, "_evicted", True)

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "live sampled signed-work pre-C registrations are nonserializable"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError(
            "live sampled signed-work pre-C registrations are nonserializable"
        )


def _validate_prepared_integrity(
    prepared: V1SampledKKTSourceProblemBundle,
) -> str:
    try:
        prepared_hash = prepared.assert_integrity()
        source = prepared.sampled_geometry_source
        admission = source.admission
        admission.moments.assert_integrity()
        admission.fd_snapshot.assert_integrity()
        admission.fd_design.assert_integrity()
        source.source_hash
    except BaseException as error:
        raise V1SampledSignedWorkPreCBridgeRefusal(
            V1SampledSignedWorkPreCBridgeRefusalCode.PREPARED_BUNDLE_MUTATED,
            "sampled KKT source problem failed deep integrity replay",
        ) from error
    problem = prepared.problem
    if (
        _problem_hash(problem) != problem.problem_hash
        or problem.source_geometry_hash != prepared.chart_identity_hash
        or problem.route_axis_hash != prepared.physical_route_axis.route_axis_hash
        or problem.route_count != prepared.physical_route_axis.route_count
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.PREPARED_BUNDLE_MUTATED,
            "prepared relaxed problem differs from its registered source bundle",
        )
    return prepared_hash


def _validate_same_d_lineage(
    *,
    prepared: V1SampledKKTSourceProblemBundle,
    restoration: V1SampledSignedWorkRowBankRestorationResult,
    original: V1SampledLiteralJOriginalCaptureBinding,
    transaction: V1SampledCompositionRegisteredDTransaction,
) -> dict[str, object]:
    prepared_hash = _validate_prepared_integrity(prepared)
    try:
        original_hash = original.assert_integrity()
        restoration_hash = restoration.receipt.assert_integrity()
        custody = restoration.row_bank_handle.receipt
        custody.assert_integrity()
    except BaseException as error:
        raise V1SampledSignedWorkPreCBridgeRefusal(
            V1SampledSignedWorkPreCBridgeRefusalCode.ORIGINAL_CAPTURE_MISMATCH,
            "original binding or restored custody failed integrity replay",
        ) from error
    rr = restoration.receipt
    if rr.original_capture_binding_hash != original_hash:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.ORIGINAL_CAPTURE_MISMATCH,
            "restored rows do not name the exact supplied original binding",
        )
    if (
        rr.row_prefix_manifest_hash != original.row_prefix_manifest_hash
        or rr.registered_d_prefix_hash != original.registered_d_prefix_hash
        or custody.registered_d_prefix_hash != original.registered_d_prefix_hash
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.STALE_REGISTERED_D_PREFIX,
            "row-prefix or registered-D prefix changed before pre-C registration",
        )
    expected_raw_prefix = transaction.rank_invariant_raw_prefix_hash
    if rr.rank_invariant_raw_prefix_hash != expected_raw_prefix:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.STALE_RAW_PREFIX,
            "restored corrected-A rows do not name the physical transaction prefix",
        )

    admission = prepared.sampled_geometry_source.admission
    source_receipt = admission.receipt
    moments = admission.moments
    snapshot = admission.fd_snapshot
    dual = admission.dual_chart_stream_binding
    if dual is None:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.STREAM_ROLE_MISMATCH,
            "sampled geometry lacks canonical corrected-A/literal-J provenance",
        )
    corrected_stream, literal_j_stream, dual_proof_hash = dual
    if (
        original.corrected_a_stream_identity_hash != corrected_stream
        or rr.corrected_a_stream_identity_hash != corrected_stream
        or original.literal_j_stream_identity_hash != literal_j_stream
        or rr.literal_j_stream_identity_hash != literal_j_stream
        or corrected_stream == literal_j_stream
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.STREAM_ROLE_MISMATCH,
            "corrected-A and literal-J roles differ from sampled geometry admission",
        )
    current_mask_hash = admission.current_mask.mask_hash
    if (
        original.chart_identity_hash != rr.chart_identity_hash
        or custody.chart_identity_hash != rr.chart_identity_hash
        or original.current_mask_hash != current_mask_hash
        or rr.current_mask_hash != current_mask_hash
        or original.proposal_generation != source_receipt.proposal_generation
        or rr.proposal_generation != source_receipt.proposal_generation
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.CHART_OR_MASK_MISMATCH,
            "capture chart, current mask, or proposal generation is stale",
        )
    physical_axis = prepared.physical_route_axis
    if (
        original.route_axis_hash != physical_axis.route_axis_hash
        or rr.route_axis_hash != physical_axis.route_axis_hash
        or custody.route_axis_hash != physical_axis.route_axis_hash
        or source_receipt.physical_route_axis_hash != physical_axis.route_axis_hash
        or original.route_id_axis_hash != physical_axis.route_id_axis_hash
        or rr.route_id_axis_hash != physical_axis.route_id_axis_hash
        or custody.route_id_axis_hash != physical_axis.route_id_axis_hash
        or source_receipt.ordered_route_axis_hash != physical_axis.route_id_axis_hash
        or original.route_count != physical_axis.route_count
        or rr.route_count != physical_axis.route_count
        or source_receipt.route_count != physical_axis.route_count
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.ROUTE_AXIS_MISMATCH,
            "original, restored, sampled, and relaxed route axes differ",
        )

    names = moments.stratum_names
    labels = _flatten(original.packet_stratum_labels)
    if set(labels) != set(names):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.SOURCE_AXIS_MISMATCH,
            "original packet strata differ from the sampled two-stratum design",
        )
    contexts_by_stratum = _group_original_axis(
        values=original.packet_context_ids,
        labels=original.packet_stratum_labels,
        stratum_names=names,
    )
    draws_by_stratum = _group_original_axis(
        values=original.packet_draw_ids,
        labels=original.packet_stratum_labels,
        stratum_names=names,
    )
    measurements_by_stratum = _group_original_axis(
        values=original.packet_fd_source_measurement_hashes,
        labels=original.packet_stratum_labels,
        stratum_names=names,
    )
    snapshot_measurements = tuple(
        values
        for name, values in snapshot.source_measurement_hashes_by_stratum
        if name in names
    )
    snapshot_contexts = tuple(
        tuple(child.context_ids)
        for name, child in snapshot.stratum_snapshots
        if name in names
    )
    if (
        contexts_by_stratum != moments.observed_context_ids_by_stratum
        or draws_by_stratum != moments.observed_draw_ids_by_stratum
        or measurements_by_stratum != snapshot_measurements
        or contexts_by_stratum != snapshot_contexts
        or tuple(len(values) for values in contexts_by_stratum)
        != moments.stratum_counts
        or rr.context_count != sum(moments.stratum_counts)
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.SOURCE_AXIS_MISMATCH,
            "original measurement/context/draw/stratum order differs from sampled D",
        )
    reconstructed_draw_hash = content_hash(
        draws_by_stratum,
        prefix=_SAMPLED_DRAW_AXIS_PREFIX,
    )
    reconstructed_context_hash = content_hash(
        contexts_by_stratum,
        prefix=_SAMPLED_CONTEXT_AXIS_PREFIX,
    )
    reconstructed_measurement_hash = content_hash(
        tuple(zip(names, measurements_by_stratum, strict=True)),
        prefix=_SAMPLED_MEASUREMENT_AXIS_PREFIX,
    )
    original_flat_measurements = _flatten(original.packet_fd_source_measurement_hashes)
    original_measurement_hash = content_hash(
        original_flat_measurements,
        prefix=_ROW_BANK_SOURCE_AXIS_PREFIX,
    )
    if (
        reconstructed_draw_hash != source_receipt.observed_draw_axis_hash
        or reconstructed_context_hash
        != source_receipt.observed_source_context_axis_hash
        or reconstructed_measurement_hash != source_receipt.source_measurement_axis_hash
        or original_measurement_hash != rr.source_measurement_axis_hash
        or tuple(snapshot.source_measurement_hashes_by_stratum)
        != tuple(zip(names, measurements_by_stratum, strict=True))
    ):
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.SOURCE_AXIS_MISMATCH,
            "reconstructed sampled-D axis hashes do not close",
        )
    return {
        "prepared_hash": prepared_hash,
        "original_hash": original_hash,
        "restoration_hash": restoration_hash,
        "custody_hash": custody.receipt_hash,
        "physical_raw_prefix_hash": expected_raw_prefix,
        "dual_proof_hash": dual_proof_hash,
        "corrected_stream": corrected_stream,
        "literal_j_stream": literal_j_stream,
        "names": names,
        "counts": moments.stratum_counts,
        "draw_hash": reconstructed_draw_hash,
        "context_hash": reconstructed_context_hash,
        "measurement_hash": reconstructed_measurement_hash,
        "packet_stratum_hash": content_hash(
            original.packet_stratum_labels,
            prefix="partition-free-v1-sampled-signed-work-original-stratum-axis-",
        ),
        "packet_measurement_hash": original_measurement_hash,
    }


def bind_v1_sampled_signed_work_pre_c_source(
    *,
    prepared: V1SampledKKTSourceProblemBundle,
    registered_d_transaction: V1SampledCompositionRegisteredDTransaction,
    restoration: V1SampledSignedWorkRowBankRestorationResult,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
) -> V1SampledSignedWorkPreCRegistrationResult:
    """Consume one restored corrected-A bank into an exact same-D pre-C source."""

    operator: V1SignedWorkOperatorCUDA | None = None
    if type(restoration) is not V1SampledSignedWorkRowBankRestorationResult:
        _refuse(
            V1SampledSignedWorkPreCBridgeRefusalCode.INVALID_TYPE,
            "pre-C binder requires the exact live restoration result",
        )
    try:
        if type(prepared) is not V1SampledKKTSourceProblemBundle:
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.INVALID_TYPE,
                "pre-C binder requires the exact sampled KKT source problem",
            )
        if (
            type(registered_d_transaction)
            is not V1SampledCompositionRegisteredDTransaction
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.INVALID_TYPE,
                "pre-C binder requires the exact physical registered-D transaction",
            )
        if (
            type(original_capture_binding)
            is not V1SampledLiteralJOriginalCaptureBinding
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.INVALID_TYPE,
                "pre-C binder requires the exact original capture binding",
            )
        lineage = _validate_same_d_lineage(
            prepared=prepared,
            restoration=restoration,
            original=original_capture_binding,
            transaction=registered_d_transaction,
        )
        crosswalk = _validate_registered_d_transaction_crosswalk(
            prepared=prepared,
            transaction=registered_d_transaction,
        )
        rr = restoration.receipt
        production = rr.production_cuda_proposal_input_authority
        design = (
            V1SignedWorkOperatorCUDAExecutionDesign()
            if production
            else V1SignedWorkOperatorCUDAExecutionDesign(
                production_cuda_required=False,
                allow_private_cpu_reference=True,
            )
        )
        operator_kwargs = dict(restoration.signed_work_operator_kwargs())
        # This override is lawful only because every same-D check above closed.
        operator_kwargs["source_geometry_hash"] = prepared.problem.source_geometry_hash
        operator_kwargs["design"] = design
        operator = V1SignedWorkOperatorCUDA.from_row_bank_device_handle(
            **operator_kwargs
        )
        if (
            operator.receipt.source_geometry_hash
            != prepared.problem.source_geometry_hash
            or operator.receipt.route_axis_hash != prepared.problem.route_axis_hash
            or operator.receipt.route_count != prepared.problem.route_count
            or operator.receipt.row_spool_receipt_hash
            != rr.row_spool_custody_receipt_hash
        ):
            _refuse(
                V1SampledSignedWorkPreCBridgeRefusalCode.SOURCE_GEOMETRY_MISMATCH,
                "constructed operator did not adopt the exact prepared geometry",
            )
        receipt = V1SampledSignedWorkPreCJointRegistrationReceipt(
            prepared_source_problem_hash=lineage["prepared_hash"],
            sampled_geometry_source_hash=prepared.sampled_geometry_source.source_hash,
            source_admission_hash=(
                prepared.sampled_geometry_source.admission.admission_hash
            ),
            response_geometry_problem_hash=prepared.problem.problem_hash,
            response_geometry_source_hash=prepared.problem.source_geometry_hash,
            original_capture_binding_hash=lineage["original_hash"],
            restoration_receipt_hash=lineage["restoration_hash"],
            row_spool_custody_receipt_hash=lineage["custody_hash"],
            row_prefix_manifest_hash=rr.row_prefix_manifest_hash,
            registered_d_prefix_hash=rr.registered_d_prefix_hash,
            registered_d_coordinate_hash=prepared.registered_d_coordinate_hash,
            physical_d_transaction_hash=crosswalk["transaction_hash"],
            physical_rank_invariant_raw_prefix_hash=(
                crosswalk["physical_raw_prefix_hash"]
            ),
            physical_literal_j_source_receipt_hash=(
                crosswalk["literal_j_source_receipt_hash"]
            ),
            sampled_observation_prefix_hash=(
                crosswalk["sampled_observation_prefix_hash"]
            ),
            selected_physical_fd_rank=crosswalk["selected_rank"],
            selected_fd_design_hash=crosswalk["selected_fd_design_hash"],
            registered_sequential_acquisition_design_hash=(
                crosswalk["acquisition_design_hash"]
            ),
            registered_look_slice_hashes=crosswalk["slice_hashes"],
            registered_look_slice_axis_hash=crosswalk["slice_axis_hash"],
            physical_to_sampled_observation_crosswalk_hash=(
                crosswalk["crosswalk_hash"]
            ),
            original_capture_chart_identity_hash=rr.chart_identity_hash,
            current_mask_hash=rr.current_mask_hash,
            proposal_generation=rr.proposal_generation,
            route_axis_hash=rr.route_axis_hash,
            route_id_axis_hash=rr.route_id_axis_hash,
            route_count=rr.route_count,
            context_count=rr.context_count,
            stratum_names=lineage["names"],
            observed_stratum_counts=lineage["counts"],
            reconstructed_draw_axis_hash=lineage["draw_hash"],
            reconstructed_source_context_axis_hash=lineage["context_hash"],
            reconstructed_source_measurement_axis_hash=lineage["measurement_hash"],
            original_packet_stratum_axis_hash=lineage["packet_stratum_hash"],
            original_packet_source_measurement_axis_hash=(
                lineage["packet_measurement_hash"]
            ),
            dual_chart_receipt_hash=lineage["dual_proof_hash"],
            corrected_a_stream_identity_hash=lineage["corrected_stream"],
            literal_j_stream_identity_hash=lineage["literal_j_stream"],
            signed_work_operator_receipt_hash=operator.receipt_hash,
            signed_work_operator_design_hash=operator.design.design_hash,
            production_cuda_proposal_input_authority=production,
            private_cpu_reference_test_only=not production,
            _issuance_token=_ISSUANCE_TOKEN,
        )
        source_bundle = (
            _issue_v1_signed_work_pre_c_source_bundle_from_sampled_registration(
                response_geometry=prepared.problem,
                signed_work_operator=operator,
                registered_d_coordinate_hash=prepared.registered_d_coordinate_hash,
                registration_receipt=receipt,
            )
            if production
            else _issue_v1_signed_work_pre_c_source_bundle_private_cpu_reference_for_tests(
                response_geometry=prepared.problem,
                signed_work_operator=operator,
                source_registration_hash=receipt.receipt_hash,
                registered_d_coordinate_hash=prepared.registered_d_coordinate_hash,
            )
        )
        result = V1SampledSignedWorkPreCRegistrationResult(
            registration_receipt=receipt,
            source_bundle=source_bundle,
            restoration_result=restoration,
            _issuance_token=_RESULT_TOKEN,
        )
        result.assert_integrity()
        return result
    except BaseException as error:
        try:
            _evict_live_sources(restoration, operator)
        except V1SampledSignedWorkPreCBridgeRefusal:
            raise
        if isinstance(error, V1SampledSignedWorkPreCBridgeRefusal):
            raise
        raise V1SampledSignedWorkPreCBridgeRefusal(
            V1SampledSignedWorkPreCBridgeRefusalCode.OPERATOR_CONSTRUCTION_FAILED,
            "sampled signed-work pre-C registration failed closed",
        ) from error


__all__ = [
    "SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_AUTHORITY_SCOPE",
    "SAMPLED_SIGNED_WORK_PRE_C_REGISTRATION_SCHEMA",
    "V1SampledSignedWorkPreCBridgeRefusal",
    "V1SampledSignedWorkPreCBridgeRefusalCode",
    "V1SampledSignedWorkPreCJointRegistrationReceipt",
    "V1SampledSignedWorkPreCRegistrationResult",
    "bind_v1_sampled_signed_work_pre_c_source",
    "derive_v1_sampled_signed_work_sampled_observation_prefix_hash",
]
