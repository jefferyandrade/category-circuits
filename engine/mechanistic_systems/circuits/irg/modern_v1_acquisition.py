"""Production-shaped acquisition and corrected global queue construction.

This module is the ownership joint for genuine modern v1.  It connects
one exact complete hard mask to streamed full-route response rows, canonical
rank-8 Frequent Directions, an identified finite-intervention correction,
and one exact finite global candidate order.  All outputs remain proposal
objects: neither geometry nor queue position can commit a mask.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.contracts.measurement import MeasurementOutcome
from mechanistic_systems.contracts.trust_geometry import GlobalTrustGeometrySpec
from mechanistic_systems.core.axis_identity import (
    ConstructiveOrdinalRouteAxis,
    ordered_string_axis_hash,
)
from mechanistic_systems.geometry.trust import (
    FrequentDirectionsTrustAccumulator,
    GlobalTrustGeometryState,
    global_trust_source_row_hash,
    global_trust_stream_weight_hash,
)

from .contracts import content_hash
from .modern_v1_campaign_contract import (
    OPTION_A_FD_RANK,
    V1OptionACampaignBinding,
)
from .modern_v1_candidate_composition import (
    FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID,
    V1CandidateCompositionConstraints,
    V1CandidateCompositionResult,
    V1CandidateNumericState,
    V1FQCandidateCompositionResult,
    V1FiniteRemovalDomain,
    _selected_route_id_axis_hash,
    compose_v1_candidates_exact,
    compose_v1_candidates_in_frozen_resource_descending_order,
    issue_v1_candidate_numeric_state,
)
from .modern_v1_core import (
    V1ChartThresholds,
    V1MixedDifferenceEnvelope,
    V1ProposalChart,
    V1ProposalChartBuildContract,
    build_v1_proposal_chart,
)
from .modern_v1_fd_adapter import (
    CUDA_FD_ALGORITHM_ID,
    CUDA_FD_IMPLEMENTATION_ID,
    CUDA_GRAM_FD_ALGORITHM_ID,
    CUDA_GRAM_FD_IMPLEMENTATION_ID,
    V1FDAdapterResult,
    V1FDSourceLedger,
    adapt_global_trust_geometry_to_v1_lineage,
)
from .modern_v1_intervention_adapter import (
    V1ActionFeatureMap,
    V1AnalyticExactGateCorrectionResult,
    V1AtomCostState,
    V1FiniteInterventionDesign,
    V1InterventionalCorrectionResult,
    V1InterventionalCorrectionSpec,
    V1TangentPriceState,
    fit_identified_interventional_correction,
)
from .modern_v1_receipts import V1AcquisitionGenerationReceipt

if TYPE_CHECKING:
    from .modern_v1_sampled_acquisition_adapters import (
        V1SampledSummaryAcquisitionSource,
    )

_FD_BATCH_SCHEMA: Final = "partition_free_v1_fd_stream_batch_v1"
_FD_ACQUISITION_SCHEMA: Final = "partition_free_v1_fd_stream_acquisition_v2"
_FD_PRODUCER_ADMISSION_SCHEMA: Final = (
    "partition_free_v1_fd_producer_admission_receipt_v1"
)
_CORRECTION_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_interventional_correction_acquisition_v1"
)
_ANALYTIC_EXACT_GATE_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_analytic_exact_gate_correction_acquisition_v1"
)
_ACQUISITION_RESULT_SCHEMA: Final = "partition_free_v1_relinearized_acquisition_v1"
_FQ_AUTHORIZED_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_fq_authorized_acquisition_v1"
)
_FQ_ACQUISITION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fq_acquisition_generation_receipt_v1"
)
_SAMPLED_DIAGNOSTIC_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_sampled_summary_diagnostic_acquisition_v1"
)
_SAMPLED_CORRECTED_POINT_LINEAGE_SCHEMA: Final = (
    "partition_free_v1_sampled_corrected_point_numeric_lineage_v1"
)
_SAMPLED_DIAGNOSTIC_AUTHORITY_SCOPE: Final = (
    "sampled_point_and_geometry_diagnostic_only_requires_selected_shell_fq_"
    "before_candidate_composition_no_oracle_mask_or_terminal_authority_v1"
)
SAMPLED_FQ_COMPOSITION_GATE_POLICY_ID: Final = (
    "selected_shell_fq_commit_plus_separate_atom_cost_binding_required_v1"
)
_SAMPLED_DIAGNOSTIC_RESERVED_METADATA_KEYS: Final = frozenset(
    {
        "authority_scope",
        "bce_oracle_authority",
        "calibration_identity",
        "campaign_binding_hash",
        "campaign_contract_hash",
        "candidate_composition_authority",
        "candidate_freeze_authority",
        "census_authority",
        "current_mask_hash",
        "exact_tangent_price_acquisition_compatible",
        "fd_stream_acquisition_compatible",
        "fq_certification_authority",
        "generation_index",
        "mask_commit_authority",
        "ordered_route_axis_hash",
        "physical_route_axis_hash",
        "proposal_only",
        "raw_logit_coordinate",
        "raw_logit_stream_source_hash",
        "response_exactification_replay_receipt_hash",
        "route_count",
        "sampled_summary_source_hash",
        "schema",
        "terminal_authority",
        "validation_inputs_present",
    }
)
_CUDA_FD_LINEAGE_IMPLEMENTATION_BY_ALGORITHM: Final = MappingProxyType(
    {
        CUDA_FD_ALGORITHM_ID: CUDA_FD_IMPLEMENTATION_ID,
        CUDA_GRAM_FD_ALGORITHM_ID: CUDA_GRAM_FD_IMPLEMENTATION_ID,
    }
)

FDBackend = Literal["canonical_cpu", "torch_cuda"]
ResidualDiagonalStatus = Literal[
    "absent",
    "proposal_only",
    "global_loewner_bound",
    "action_class_quadratic_bound",
]


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


def _bytes_f64(array: np.ndarray, *, name: str) -> tuple[bytes, tuple[int, ...]]:
    values = np.asarray(array, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite and nonempty")
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    return contiguous.tobytes(order="C"), tuple(int(value) for value in contiguous.shape)


def _aligned_strings(
    values: Sequence[str],
    label: str,
    *,
    unique: bool,
) -> Sequence[str]:
    if unique and isinstance(values, ConstructiveOrdinalRouteAxis):
        # The route:<decimal ordinal> grammar is injective by construction.
        # Preserve this O(1)-storage axis instead of expanding it into both a
        # many-million-element tuple and a uniqueness-validation set.
        return values
    result = tuple(_nonempty(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must be nonempty")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class V1FDStreamBatch:
    """Immutable packet of logical FD rows and their physical-source lineage."""

    batch_id: str
    draw_ids: tuple[str, ...]
    source_context_ids: tuple[str, ...]
    rows_payload: bytes = field(repr=False)
    row_shape: tuple[int, int, int]
    weights_payload: bytes = field(repr=False)
    source_measurement_hash: str
    request_hash: str
    provider_hash: str
    calibration_identity: str
    current_mask_hash: str
    route_axis_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _FD_BATCH_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "batch_id",
            "source_measurement_hash",
            "request_hash",
            "provider_hash",
            "calibration_identity",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        draws = _aligned_strings(self.draw_ids, "draw_ids", unique=True)
        sources = _aligned_strings(
            self.source_context_ids,
            "source_context_ids",
            unique=False,
        )
        if len(draws) != len(sources):
            raise ValueError("FD draw/source axes must align one-to-one")
        shape = tuple(self.row_shape)
        if (
            len(shape) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
                for value in shape
            )
        ):
            raise ValueError("FD row_shape must be positive [draw,response,route]")
        normalized_shape = tuple(int(value) for value in shape)
        if normalized_shape[0] != len(draws):
            raise ValueError("FD row_shape draw axis differs from draw_ids")
        if type(self.rows_payload) is not bytes or len(self.rows_payload) != (
            8 * math.prod(normalized_shape)
        ):
            raise ValueError("FD rows_payload is not aligned immutable float64")
        if type(self.weights_payload) is not bytes or len(self.weights_payload) != 8 * len(draws):
            raise ValueError("FD weights_payload is not aligned immutable float64")
        if not np.isfinite(self.rows).all():
            raise ValueError("FD rows contain nonfinite values")
        weights = self.weights
        if not np.isfinite(weights).all() or bool(np.any(weights < 0.0)) or float(weights.sum()) <= 0.0:
            raise ValueError("FD weights must be finite, nonnegative, and nonzero")
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "source_context_ids", sources)
        object.__setattr__(self, "row_shape", normalized_shape)
        if self.schema != _FD_BATCH_SCHEMA:
            raise ValueError("unsupported modern-v1 FD stream-batch schema")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "FD batch metadata"))

    @classmethod
    def create(
        cls,
        *,
        batch_id: str,
        draw_ids: Sequence[str],
        source_context_ids: Sequence[str],
        rows: np.ndarray,
        weights: Sequence[float] | np.ndarray,
        source_measurement_hash: str,
        request_hash: str,
        provider_hash: str,
        calibration_identity: str,
        current_mask_hash: str,
        route_axis_hash: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "V1FDStreamBatch":
        rows_payload, row_shape = _bytes_f64(np.asarray(rows), name="FD rows")
        weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
        weights_payload, weight_shape = _bytes_f64(weight_array, name="FD weights")
        if weight_shape != (row_shape[0],):
            raise ValueError("FD weights must align one-to-one with draws")
        return cls(
            batch_id=batch_id,
            draw_ids=tuple(draw_ids),
            source_context_ids=tuple(source_context_ids),
            rows_payload=rows_payload,
            row_shape=row_shape,  # type: ignore[arg-type]
            weights_payload=weights_payload,
            source_measurement_hash=source_measurement_hash,
            request_hash=request_hash,
            provider_hash=provider_hash,
            calibration_identity=calibration_identity,
            current_mask_hash=current_mask_hash,
            route_axis_hash=route_axis_hash,
            metadata={} if metadata is None else metadata,
        )

    @property
    def rows(self) -> np.ndarray:
        return np.frombuffer(self.rows_payload, dtype="<f8").reshape(self.row_shape)

    @property
    def weights(self) -> np.ndarray:
        return np.frombuffer(self.weights_payload, dtype="<f8")

    @property
    def batch_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fd-stream-batch-")


@dataclass(frozen=True, slots=True)
class V1FDProducerAdmissionReceipt:
    """Typed admission of one exact bounded producer-owned FD state.

    The receipt is issued only after the canonical modern-v1 adapter has
    recomputed the accumulator algebra from the immutable state and exact
    draw/source ledger.  It admits a proposal source; it never authorizes a
    decision, mask transition, selector, or scientific claim.
    """

    campaign_contract_hash: str
    campaign_binding_hash: str
    current_mask_hash: str
    route_axis_hash: str
    backend: FDBackend
    source_state_hash: str
    source_ledger_hash: str
    accumulator_derivation_receipt_hash: str
    adapter_result_hash: str
    derivation_proof_mode: str
    proof_streaming_window_contexts: int
    proof_peak_host_row_bytes: int
    proof_total_host_transfer_bytes: int
    full_tensor_host_snapshot: bool = False
    proposal_only: bool = True
    claim_execution_certified: bool = False
    schema: str = _FD_PRODUCER_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "current_mask_hash",
            "route_axis_hash",
            "source_state_hash",
            "source_ledger_hash",
            "accumulator_derivation_receipt_hash",
            "adapter_result_hash",
            "derivation_proof_mode",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.backend != "torch_cuda":
            raise ValueError("bounded producer admission currently requires torch_cuda")
        if self.derivation_proof_mode != "bounded_ordered_row_stream_exact_v1":
            raise ValueError("production CUDA admission requires the bounded exact source proof")
        object.__setattr__(
            self,
            "proof_streaming_window_contexts",
            _positive_int(
                self.proof_streaming_window_contexts,
                "proof_streaming_window_contexts",
            ),
        )
        if self.proof_streaming_window_contexts != 1:
            raise ValueError("bounded CUDA proof window must contain exactly one logical row")
        object.__setattr__(
            self,
            "proof_peak_host_row_bytes",
            _positive_int(self.proof_peak_host_row_bytes, "proof_peak_host_row_bytes"),
        )
        object.__setattr__(
            self,
            "proof_total_host_transfer_bytes",
            _positive_int(
                self.proof_total_host_transfer_bytes,
                "proof_total_host_transfer_bytes",
            ),
        )
        if self.proof_total_host_transfer_bytes < self.proof_peak_host_row_bytes:
            raise ValueError("bounded CUDA proof transfer accounting is inconsistent")
        if type(self.full_tensor_host_snapshot) is not bool or self.full_tensor_host_snapshot:
            raise ValueError("bounded CUDA producer admission forbids a full host tensor snapshot")
        if type(self.proposal_only) is not bool or not self.proposal_only:
            raise ValueError("FD producer admission is proposal-only")
        if type(self.claim_execution_certified) is not bool or self.claim_execution_certified:
            raise ValueError("FD producer admission cannot certify execution")
        if self.schema != _FD_PRODUCER_ADMISSION_SCHEMA:
            raise ValueError("unsupported FD producer-admission schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fd-producer-admission-")


@dataclass(frozen=True, slots=True)
class V1FDStreamAcquisition:
    """Verified full-axis FD state and modern-v1 adapter lineage."""

    campaign_contract_hash: str
    current_mask_hash: str
    route_axis_hash: str
    backend: FDBackend
    batches: tuple[V1FDStreamBatch, ...]
    geometry_state: GlobalTrustGeometryState
    source_ledger: V1FDSourceLedger
    adapter_result: V1FDAdapterResult
    producer_admission_receipt: V1FDProducerAdmissionReceipt | None = None
    schema: str = _FD_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("campaign_contract_hash", "current_mask_hash", "route_axis_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.backend not in {"canonical_cpu", "torch_cuda"}:
            raise ValueError("unsupported modern-v1 FD backend")
        batches = tuple(self.batches)
        if any(type(batch) is not V1FDStreamBatch for batch in batches):
            raise TypeError("FD acquisition batches must be exact immutable stream batches")
        producer_receipt = self.producer_admission_receipt
        if self.backend == "canonical_cpu":
            if not batches:
                raise TypeError("canonical FD acquisition requires immutable stream batches")
            if producer_receipt is not None:
                raise ValueError("canonical FD acquisition cannot carry a CUDA producer receipt")
        else:
            if batches:
                raise ValueError("CUDA FD admission must not retain raw host stream batches")
            if type(producer_receipt) is not V1FDProducerAdmissionReceipt:
                raise TypeError("CUDA FD acquisition requires an exact producer-admission receipt")
        if type(self.geometry_state) is not GlobalTrustGeometryState:
            raise TypeError("FD acquisition requires an exact global trust state")
        if type(self.source_ledger) is not V1FDSourceLedger:
            raise TypeError("FD acquisition requires an exact source ledger")
        if type(self.adapter_result) is not V1FDAdapterResult:
            raise TypeError("FD acquisition requires an exact adapter result")
        if (
            self.geometry_state.measured_mask_hash != self.current_mask_hash
            or self.geometry_state.derivation_receipt is None
            or self.source_ledger.expected_measured_mask_hash != self.current_mask_hash
            or self.adapter_result.lineage.measured_mask_hash != self.current_mask_hash
        ):
            raise ValueError("FD acquisition is not measured at the exact current mask")
        if (
            self.source_ledger.expected_ordered_route_axis_hash != self.route_axis_hash
            or self.adapter_result.lineage.route_axis_hash != self.route_axis_hash
        ):
            raise ValueError("FD acquisition route-axis lineage differs")
        if self.adapter_result.lineage.metadata.get("centered_fd_sketch_rank") != OPTION_A_FD_RANK:
            raise ValueError("modern-v1 production acquisition requires centered FD rank 8")
        if self.backend == "canonical_cpu":
            expected_implementation = "canonical_numpy"
        else:
            derivation = self.geometry_state.derivation_receipt
            if derivation is None:  # defensive: validated above
                raise ValueError("CUDA FD acquisition lacks a derivation receipt")
            expected_implementation = _CUDA_FD_LINEAGE_IMPLEMENTATION_BY_ALGORITHM.get(
                derivation.producer_algorithm_id
            )
            if expected_implementation is None:
                raise ValueError("CUDA FD acquisition names an unsupported producer algorithm")
        if self.adapter_result.lineage.implementation != expected_implementation:
            raise ValueError("FD backend differs from the verified producer implementation")
        if producer_receipt is not None:
            derivation = self.geometry_state.derivation_receipt
            if derivation is None:  # defensive: validated above, never optimize away
                raise ValueError("FD producer admission requires a derivation receipt")
            if (
                producer_receipt.campaign_contract_hash != self.campaign_contract_hash
                or producer_receipt.current_mask_hash != self.current_mask_hash
                or producer_receipt.route_axis_hash != self.route_axis_hash
                or producer_receipt.backend != self.backend
                or producer_receipt.source_state_hash != self.geometry_state.content_hash
                or producer_receipt.source_ledger_hash != self.source_ledger.ledger_hash
                or producer_receipt.accumulator_derivation_receipt_hash
                != derivation.receipt_hash
                or producer_receipt.adapter_result_hash != self.adapter_result.result_hash
            ):
                raise ValueError("FD producer-admission receipt differs from admitted content")
        if self.schema != _FD_ACQUISITION_SCHEMA:
            raise ValueError("unsupported FD acquisition schema")
        object.__setattr__(self, "batches", batches)

    @property
    def acquisition_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fd-stream-acquisition-")


@dataclass(frozen=True, slots=True)
class V1CorrectionAcquisition:
    """Identified finite-intervention correction bound to one current mask."""

    campaign_contract_hash: str
    current_mask_hash: str
    route_axis_hash: str
    feature_map: V1ActionFeatureMap
    tangent_prices: V1TangentPriceState
    atom_costs: V1AtomCostState
    design: V1FiniteInterventionDesign
    specification: V1InterventionalCorrectionSpec
    outcomes: tuple[MeasurementOutcome, ...]
    result: V1InterventionalCorrectionResult
    schema: str = _CORRECTION_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("campaign_contract_hash", "current_mask_hash", "route_axis_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        exact_types = (
            (self.feature_map, V1ActionFeatureMap, "feature_map"),
            (self.tangent_prices, V1TangentPriceState, "tangent_prices"),
            (self.atom_costs, V1AtomCostState, "atom_costs"),
            (self.design, V1FiniteInterventionDesign, "design"),
            (self.specification, V1InterventionalCorrectionSpec, "specification"),
            (self.result, V1InterventionalCorrectionResult, "result"),
        )
        for value, expected, label in exact_types:
            if type(value) is not expected:
                raise TypeError(f"correction acquisition requires exact {label}")
        outcomes = tuple(self.outcomes)
        if not outcomes or any(type(outcome) is not MeasurementOutcome for outcome in outcomes):
            raise TypeError("correction acquisition requires exact provider outcomes")
        route_hashes = {
            self.feature_map.route_axis_hash,
            self.tangent_prices.route_axis_hash,
            self.atom_costs.route_axis_hash,
            self.result.lineage.route_axis_hash,
        }
        if route_hashes != {self.route_axis_hash}:
            raise ValueError("correction acquisition route axes differ")
        if (
            self.tangent_prices.base_mask_hash != self.current_mask_hash
            or self.design.base_mask_hash != self.current_mask_hash
            or self.result.lineage.base_mask_hash != self.current_mask_hash
        ):
            raise ValueError("correction acquisition is not identified at the current mask")
        if self.schema != _CORRECTION_ACQUISITION_SCHEMA:
            raise ValueError("unsupported correction acquisition schema")
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def acquisition_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-correction-acquisition-")


@dataclass(frozen=True, slots=True)
class V1AnalyticExactGateCorrectionAcquisition:
    """Canonical no-mask-evaluation correction from exact gate rows."""

    campaign_contract_hash: str
    current_mask_hash: str
    route_axis_hash: str
    context_mixture_hash: str
    tangent_prices: V1TangentPriceState
    atom_costs: V1AtomCostState
    result: V1AnalyticExactGateCorrectionResult
    source_measurement_hashes: tuple[str, ...]
    outcomes: tuple[()] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _ANALYTIC_EXACT_GATE_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "current_mask_hash",
            "route_axis_hash",
            "context_mixture_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.tangent_prices) is not V1TangentPriceState:
            raise TypeError("analytic exact-gate acquisition requires tangent prices")
        if type(self.atom_costs) is not V1AtomCostState:
            raise TypeError("analytic exact-gate acquisition requires atom costs")
        if type(self.result) is not V1AnalyticExactGateCorrectionResult:
            raise TypeError("analytic exact-gate acquisition requires its exact result")
        lineage = self.result.lineage
        if (
            self.tangent_prices.route_axis_hash != self.route_axis_hash
            or self.atom_costs.route_axis_hash != self.route_axis_hash
            or lineage.route_axis_hash != self.route_axis_hash
            or self.tangent_prices.base_mask_hash != self.current_mask_hash
            or lineage.base_mask_hash != self.current_mask_hash
            or self.tangent_prices.context_mixture_hash != self.context_mixture_hash
            or lineage.context_mixture_hash != self.context_mixture_hash
            or self.tangent_prices.state_hash != lineage.tangent_price_state_hash
            or self.atom_costs.state_hash != lineage.atom_cost_receipt_hash
            or not np.array_equal(
                self.tangent_prices.prices,
                self.result.corrected_prices,
            )
        ):
            raise ValueError("analytic exact-gate acquisition sources do not close")
        measurements = tuple(
            _nonempty(value, "source_measurement_hash")
            for value in self.source_measurement_hashes
        )
        if not measurements or len(set(measurements)) != len(measurements):
            raise ValueError("analytic exact-gate source measurements must be unique")
        if tuple(self.outcomes):
            raise ValueError("analytic exact-gate acquisition forbids oracle outcomes")
        if self.schema != _ANALYTIC_EXACT_GATE_ACQUISITION_SCHEMA:
            raise ValueError("unsupported analytic exact-gate acquisition schema")
        object.__setattr__(self, "source_measurement_hashes", measurements)
        object.__setattr__(self, "outcomes", ())
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))

    @property
    def acquisition_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-analytic-exact-gate-acquisition-",
        )


@dataclass(frozen=True, slots=True)
class V1RelinearizedAcquisition:
    """Complete fresh chart, exact finite domain, and deterministic global queue."""

    binding: V1OptionACampaignBinding
    generation_index: int
    current_mask: HardMask
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    fd: V1FDStreamAcquisition
    correction: (
        V1CorrectionAcquisition
        | V1AnalyticExactGateCorrectionAcquisition
    )
    chart: V1ProposalChart
    numeric_state: V1CandidateNumericState
    domain: V1FiniteRemovalDomain
    constraints: V1CandidateCompositionConstraints
    composition: V1CandidateCompositionResult
    receipt: V1AcquisitionGenerationReceipt
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _ACQUISITION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("relinearized acquisition requires an exact campaign binding")
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, "generation_index"),
        )
        if type(self.current_mask) is not HardMask:
            raise TypeError("relinearized acquisition requires an exact HardMask")
        for value, expected, label in (
            (self.fd, V1FDStreamAcquisition, "FD acquisition"),
            (self.chart, V1ProposalChart, "proposal chart"),
            (self.numeric_state, V1CandidateNumericState, "numeric state"),
            (self.domain, V1FiniteRemovalDomain, "finite domain"),
            (self.constraints, V1CandidateCompositionConstraints, "constraints"),
            (self.composition, V1CandidateCompositionResult, "composition"),
            (self.receipt, V1AcquisitionGenerationReceipt, "receipt"),
        ):
            if type(value) is not expected:
                raise TypeError(f"relinearized acquisition requires exact {label}")
        if type(self.correction) not in {
            V1CorrectionAcquisition,
            V1AnalyticExactGateCorrectionAcquisition,
        }:
            raise TypeError("relinearized acquisition requires exact correction acquisition")
        lineage = tuple(self.committed_mask_lineage_hashes)
        transitions = tuple(self.committed_mask_transition_receipt_hashes)
        if (
            not lineage
            or lineage[-1] != self.current_mask.mask_hash
            or len(transitions) != len(lineage) - 1
        ):
            raise ValueError("relinearized acquisition has incomplete exact mask ancestry")
        if len(set(lineage)) != len(lineage) or len(set(transitions)) != len(transitions):
            raise ValueError("relinearized mask ancestry contains duplicate identities")
        if (
            self.fd.current_mask_hash != self.current_mask.mask_hash
            or self.correction.current_mask_hash != self.current_mask.mask_hash
            or self.chart.current_mask_hash != self.current_mask.mask_hash
            or self.chart.base_mask_hash != self.current_mask.mask_hash
            or self.numeric_state.parent_mask_hash != self.current_mask.mask_hash
        ):
            raise ValueError("relinearized acquisition contains a stale current mask")
        if self.composition.numeric_state.state_hash != self.numeric_state.state_hash:
            raise ValueError("composition names a different numeric state")
        if self.composition.domain.domain_hash != self.domain.domain_hash:
            raise ValueError("composition names a different finite domain")
        if self.composition.constraints.constraints_hash != self.constraints.constraints_hash:
            raise ValueError("composition names different constraints")
        if tuple(item.candidate_id for item in self.composition.ranked_proposals) != (
            self.receipt.ordered_candidate_ids
        ):
            raise ValueError("acquisition receipt candidate order differs")
        if self.receipt.acquisition_hash != self.acquisition_hash:
            raise ValueError("acquisition receipt names different content")
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(self, "committed_mask_transition_receipt_hashes", transitions)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "acquisition metadata"))
        if self.schema != _ACQUISITION_RESULT_SCHEMA:
            raise ValueError("unsupported relinearized acquisition schema")

    @property
    def acquisition_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "campaign_binding_hash": self.binding.binding_hash,
                "generation_index": self.generation_index,
                "current_mask_hash": self.current_mask.mask_hash,
                "committed_mask_lineage_hashes": self.committed_mask_lineage_hashes,
                "committed_mask_transition_receipt_hashes": (
                    self.committed_mask_transition_receipt_hashes
                ),
                "fd_acquisition_hash": self.fd.acquisition_hash,
                "correction_acquisition_hash": self.correction.acquisition_hash,
                "chart_hash": self.chart.chart_hash,
                "numeric_state_hash": self.numeric_state.state_hash,
                "domain_hash": self.domain.domain_hash,
                "constraints_hash": self.constraints.constraints_hash,
                "composition_result_hash": self.composition.result_hash,
                "metadata": self.metadata,
            },
            prefix="partition-free-v1-relinearized-acquisition-",
        )

    @property
    def queue(self) -> tuple:
        return self.composition.ranked_proposals

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


def _v1_fq_invalidation_chart_hash(
    *,
    generation_index: int,
    current_mask: HardMask,
    composition: V1FQCandidateCompositionResult,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_fq_invalidation_chart_v1",
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "route_axis_hash": current_mask.route_axis_hash,
            "sampled_diagnostic_lineage_hash": (
                composition.sampled_diagnostic_lineage_hash
            ),
            "fq_queue_authorization_hash": composition.fq_queue_authorization_hash,
            "decisive_fq_commit_hash": composition.decisive_fq_commit_hash,
            "fq_budget_hash": composition.fq_budget_hash,
            "atom_cost_source_hash": composition.atom_cost_source_hash,
        },
        prefix="partition-free-v1-fq-invalidation-chart-",
    )


def _v1_fq_authorized_acquisition_hash(
    *,
    schema: str,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: tuple[str, ...],
    committed_mask_transition_receipt_hashes: tuple[str, ...],
    invalidation_chart_hash: str,
    composition_result_hash: str,
    metadata: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "schema": schema,
            "campaign_binding_hash": binding.binding_hash,
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "committed_mask_lineage_hashes": committed_mask_lineage_hashes,
            "committed_mask_transition_receipt_hashes": (
                committed_mask_transition_receipt_hashes
            ),
            "invalidation_chart_hash": invalidation_chart_hash,
            "composition_result_hash": composition_result_hash,
            "metadata": metadata,
        },
        prefix="partition-free-v1-fq-authorized-acquisition-",
    )


@dataclass(frozen=True, slots=True)
class V1FQAcquisitionGenerationReceipt:
    """Compact receipt for one selected-shell F/Q-authorized queue.

    This receipt deliberately does not reuse the exact-stream acquisition
    receipt's FD, correction, chart, or finite-domain fields.  Its scientific
    sources are the sampled diagnostic lineage, decisive selected-shell F/Q
    commit, registered F/Q budget, and separately bound atom-cost source.
    """

    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    acquisition_hash: str
    invalidation_chart_hash: str
    composition_result_hash: str
    compact_domain_hash: str
    fq_constraints_hash: str
    fq_queue_authorization_hash: str
    decisive_fq_commit_hash: str
    fq_budget_hash: str
    fq_certification_receipt_hash: str
    sampled_diagnostic_lineage_hash: str
    atom_cost_source_hash: str
    ordered_candidate_ids: tuple[str, ...]
    ordered_endpoint_indices: tuple[int, ...]
    ordered_action_hashes: tuple[str, ...]
    refresh_reason: str
    logical_sequence: int
    proposal_only: bool = True
    exact_registered_surrogate_evaluation: bool = False
    exact_intervention_damage_evaluated: bool = False
    candidate_freeze_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    claim_execution_certified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _FQ_ACQUISITION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
            "current_mask_hash",
            "route_axis_hash",
            "acquisition_hash",
            "invalidation_chart_hash",
            "composition_result_hash",
            "compact_domain_hash",
            "fq_constraints_hash",
            "fq_queue_authorization_hash",
            "decisive_fq_commit_hash",
            "fq_budget_hash",
            "fq_certification_receipt_hash",
            "sampled_diagnostic_lineage_hash",
            "atom_cost_source_hash",
            "refresh_reason",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        generation = _nonnegative_int(self.generation_index, "generation_index")
        sequence = _nonnegative_int(self.logical_sequence, "logical_sequence")
        candidate_ids = _aligned_strings(
            self.ordered_candidate_ids,
            "ordered_candidate_ids",
            unique=True,
        )
        action_hashes = _aligned_strings(
            self.ordered_action_hashes,
            "ordered_action_hashes",
            unique=True,
        )
        endpoints = tuple(
            _positive_int(value, "ordered_endpoint_index")
            for value in self.ordered_endpoint_indices
        )
        if (
            not endpoints
            or len(endpoints) != len(candidate_ids)
            or len(endpoints) != len(action_hashes)
            or len(set(endpoints)) != len(endpoints)
        ):
            raise ValueError("F/Q acquisition receipt queue axes do not align")
        firewall = (
            self.proposal_only,
            not self.exact_registered_surrogate_evaluation,
            not self.exact_intervention_damage_evaluated,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.claim_execution_certified,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("F/Q acquisition receipt crossed its proposal-only firewall")
        if self.schema != _FQ_ACQUISITION_RECEIPT_SCHEMA:
            raise ValueError("unsupported F/Q acquisition receipt schema")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "logical_sequence", sequence)
        object.__setattr__(self, "ordered_candidate_ids", candidate_ids)
        object.__setattr__(self, "ordered_endpoint_indices", endpoints)
        object.__setattr__(self, "ordered_action_hashes", action_hashes)
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, "F/Q acquisition receipt metadata"),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-fq-acquisition-generation-",
        )


@dataclass(frozen=True, slots=True)
class V1FQAuthorizedAcquisition:
    """Compact, proposal-only queue from a decisive selected-shell F/Q look.

    The full sampled summary graph and full F/Q authorization graph are
    intentionally absent.  Their immutable hashes are retained by the compact
    composition.  This keeps durable size proportional to the endpoint count
    and packed masks rather than to the physical route axis.
    """

    binding: V1OptionACampaignBinding
    generation_index: int
    current_mask: HardMask
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    composition: V1FQCandidateCompositionResult
    receipt: V1FQAcquisitionGenerationReceipt
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _FQ_AUTHORIZED_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("F/Q acquisition requires an exact campaign binding")
        generation = _nonnegative_int(self.generation_index, "generation_index")
        if type(self.current_mask) is not HardMask:
            raise TypeError("F/Q acquisition requires an exact current mask")
        if type(self.composition) is not V1FQCandidateCompositionResult:
            raise TypeError("F/Q acquisition requires its compact composition")
        if type(self.receipt) is not V1FQAcquisitionGenerationReceipt:
            raise TypeError("F/Q acquisition requires its compact receipt")
        self.composition.assert_integrity()
        lineage = tuple(self.committed_mask_lineage_hashes)
        transitions = tuple(self.committed_mask_transition_receipt_hashes)
        if (
            not lineage
            or lineage[-1] != self.current_mask.mask_hash
            or len(transitions) != len(lineage) - 1
            or len(set(lineage)) != len(lineage)
            or len(set(transitions)) != len(transitions)
        ):
            raise ValueError("F/Q acquisition has incomplete exact mask ancestry")
        contract = self.binding.contract
        composition = self.composition
        first_state = composition.action_numeric_states[0]
        if (
            contract.route_axis_hash != self.current_mask.route_axis_hash
            or composition.route_axis_hash != self.current_mask.route_axis_hash
            or composition.route_count != self.current_mask.route_count
            or composition.parent_mask_hash != self.current_mask.mask_hash
            or first_state.generation_index != generation
        ):
            raise ValueError("F/Q acquisition differs from the current generation or axis")
        receipt = self.receipt
        if (
            receipt.campaign_contract_hash != contract.campaign_contract_hash
            or receipt.campaign_binding_hash != self.binding.binding_hash
            or receipt.registration_hash != self.binding.registration.registration_hash
            or receipt.generation_index != generation
            or receipt.current_mask_hash != self.current_mask.mask_hash
            or receipt.route_axis_hash != self.current_mask.route_axis_hash
            or receipt.composition_result_hash != composition.result_hash
            or receipt.compact_domain_hash != composition.domain_hash
            or receipt.fq_constraints_hash != composition.constraints_hash
            or receipt.fq_queue_authorization_hash
            != composition.fq_queue_authorization_hash
            or receipt.decisive_fq_commit_hash != composition.decisive_fq_commit_hash
            or receipt.fq_budget_hash != composition.fq_budget_hash
            or receipt.fq_certification_receipt_hash
            != composition.fq_certification_receipt_hash
            or receipt.sampled_diagnostic_lineage_hash
            != composition.sampled_diagnostic_lineage_hash
            or receipt.atom_cost_source_hash != composition.atom_cost_source_hash
            or receipt.ordered_candidate_ids
            != tuple(item.candidate_id for item in composition.ranked_proposals)
            or receipt.ordered_endpoint_indices != composition.queue_endpoint_indices
            or receipt.ordered_action_hashes != composition.queue_action_hashes
            or receipt.invalidation_chart_hash != self.invalidation_chart_hash
            or receipt.acquisition_hash != self.acquisition_hash
        ):
            raise ValueError("F/Q acquisition receipt differs from its compact sources")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(self, "committed_mask_transition_receipt_hashes", transitions)
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, "F/Q acquisition metadata"),
        )
        if self.schema != _FQ_AUTHORIZED_ACQUISITION_SCHEMA:
            raise ValueError("unsupported F/Q authorized acquisition schema")

    @property
    def invalidation_chart_hash(self) -> str:
        return _v1_fq_invalidation_chart_hash(
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            composition=self.composition,
        )

    @property
    def acquisition_hash(self) -> str:
        return _v1_fq_authorized_acquisition_hash(
            schema=self.schema,
            binding=self.binding,
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            committed_mask_lineage_hashes=self.committed_mask_lineage_hashes,
            committed_mask_transition_receipt_hashes=(
                self.committed_mask_transition_receipt_hashes
            ),
            invalidation_chart_hash=self.invalidation_chart_hash,
            composition_result_hash=self.composition.result_hash,
            metadata=self.metadata,
        )

    @property
    def queue(self) -> tuple:
        return self.composition.ranked_proposals

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


def issue_v1_fq_authorized_acquisition(
    *,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    composition: V1FQCandidateCompositionResult,
    refresh_reason: str = "fresh_current_mask_fq_authorized_relinearization",
    logical_sequence: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> V1FQAuthorizedAcquisition:
    """Issue the compact acquisition without retaining its full source graphs."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("F/Q acquisition issuance requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("F/Q acquisition issuance requires an exact current mask")
    if type(composition) is not V1FQCandidateCompositionResult:
        raise TypeError("F/Q acquisition issuance requires its compact composition")
    composition.assert_integrity()
    generation = _nonnegative_int(generation_index, "generation_index")
    sequence = _nonnegative_int(logical_sequence, "logical_sequence")
    lineage = tuple(committed_mask_lineage_hashes)
    transitions = tuple(committed_mask_transition_receipt_hashes)
    reason = _nonempty(refresh_reason, "refresh_reason")
    if (
        not lineage
        or lineage[-1] != current_mask.mask_hash
        or len(transitions) != len(lineage) - 1
    ):
        raise ValueError("F/Q acquisition issuance has incomplete mask ancestry")
    required_metadata = {
        "refresh_reason": reason,
        "logical_sequence": sequence,
        "generation_index": generation,
        "candidate_composition_mode": (
            "fq_authorized_resource_descending_compact"
        ),
        "sampled_numeric_lineage_embedded": False,
        "full_fq_authorization_graph_embedded": False,
        "full_route_numeric_state_embedded": False,
        "fq_monotonicity_assumed": False,
        "proposal_only": True,
    }
    caller_metadata = dict(metadata or {})
    collisions = sorted(set(required_metadata).intersection(caller_metadata))
    if collisions:
        raise ValueError(
            f"caller metadata shadows reserved compact F/Q fields: {collisions}"
        )
    core_metadata = _freeze(
        {**caller_metadata, **required_metadata},
        "F/Q acquisition metadata",
    )
    invalidation_hash = _v1_fq_invalidation_chart_hash(
        generation_index=generation,
        current_mask=current_mask,
        composition=composition,
    )
    acquisition_hash = _v1_fq_authorized_acquisition_hash(
        schema=_FQ_AUTHORIZED_ACQUISITION_SCHEMA,
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        invalidation_chart_hash=invalidation_hash,
        composition_result_hash=composition.result_hash,
        metadata=core_metadata,
    )
    receipt = V1FQAcquisitionGenerationReceipt(
        campaign_contract_hash=binding.contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        registration_hash=binding.registration.registration_hash,
        generation_index=generation,
        current_mask_hash=current_mask.mask_hash,
        route_axis_hash=current_mask.route_axis_hash,
        acquisition_hash=acquisition_hash,
        invalidation_chart_hash=invalidation_hash,
        composition_result_hash=composition.result_hash,
        compact_domain_hash=composition.domain_hash,
        fq_constraints_hash=composition.constraints_hash,
        fq_queue_authorization_hash=composition.fq_queue_authorization_hash,
        decisive_fq_commit_hash=composition.decisive_fq_commit_hash,
        fq_budget_hash=composition.fq_budget_hash,
        fq_certification_receipt_hash=composition.fq_certification_receipt_hash,
        sampled_diagnostic_lineage_hash=(
            composition.sampled_diagnostic_lineage_hash
        ),
        atom_cost_source_hash=composition.atom_cost_source_hash,
        ordered_candidate_ids=tuple(
            item.candidate_id for item in composition.ranked_proposals
        ),
        ordered_endpoint_indices=composition.queue_endpoint_indices,
        ordered_action_hashes=composition.queue_action_hashes,
        refresh_reason=reason,
        logical_sequence=sequence,
        metadata={
            "validated_action_count": composition.validated_action_count,
            "proposal_count": composition.proposal_count,
            "surrogate_evaluated_action_count": 0,
            "authorized_finite_queue_materialized_exactly": True,
            "filtering_performed": False,
            "reordering_performed": False,
        },
    )
    result = V1FQAuthorizedAcquisition(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        composition=composition,
        receipt=receipt,
        metadata=core_metadata,
    )
    if result.acquisition_hash != acquisition_hash:
        raise RuntimeError("compact F/Q acquisition hash is not reproducible")
    return result


@dataclass(frozen=True, slots=True)
class V1SampledSummaryDiagnosticAcquisition:
    """Truthful sampled point/geometry owner with no exact-source upgrade.

    The sampled registered-look path retains sufficient summaries rather than
    the complete ordered response-row stream.  This object binds those
    summaries to the exact campaign, generation, current mask, coordinate,
    and response-exactification replay without pretending that the missing
    stream ledger or an exact tangent-price acquisition exists.  It is a
    diagnostic numeric source only; a later joint must additionally bind a
    decisive selected-shell F/Q commit and atom-cost state before composing
    candidates.
    """

    binding: V1OptionACampaignBinding = field(repr=False, compare=False)
    current_mask: HardMask = field(repr=False, compare=False)
    sampled_source: "V1SampledSummaryAcquisitionSource" = field(
        repr=False,
        compare=False,
    )
    generation_index: int
    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    context_population_hash: str
    sampling_law_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    raw_logit_stream_source_hash: str
    raw_logit_coordinate: str
    sampled_summary_source_hash: str
    discovery_source_receipt_hash: str
    discovery_admission_hash: str
    geometry_receipt_hash: str
    geometry_result_hash: str
    geometry_state_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    source_measurement_axis_hash: str
    corrected_moments_state_hash: str
    signed_point_hash: str
    absolute_point_hash: str
    corrected_point_estimate_hash: str
    corrected_point_status: str
    response_exactification_replay_receipt_hash: str
    score_order_registration_hash: str
    score_order_look_receipt_hash: str
    score_order_confidence_box_hash: str
    score_order_regret_receipt_axis_hash: str
    permutation_lineage_hash: str
    validation_firewall_hash: str
    terminal_validation_population_hash: str
    terminal_validation_policy_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sampled_summary_compatible: bool = True
    census_authority: bool = False
    raw_context_route_rows_retained: bool = False
    full_per_draw_source_row_hash_ledger_retained: bool = False
    source_row_proof_authority: bool = False
    exact_stream_rows_available: bool = False
    standard_fd_stream_acquisition_compatible: bool = False
    exact_tangent_price_acquisition_compatible: bool = False
    exact_candidate_numeric_state_issued: bool = False
    score_order_authority: bool = False
    fq_certification_authority: bool = False
    candidate_composition_authority: bool = False
    candidate_freeze_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    validation_inputs_present: bool = False
    proposal_only: bool = True
    authority_scope: str = _SAMPLED_DIAGNOSTIC_AUTHORITY_SCOPE
    schema: str = _SAMPLED_DIAGNOSTIC_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        # Keep the sampled adapter behind a local runtime boundary.  This
        # avoids making the exact acquisition module import GPU/sample-path
        # dependencies during ordinary exact-stream use and remains safe if
        # the sampled adapter later consumes public acquisition types.
        from .modern_v1_sampled_acquisition_adapters import (
            SAMPLED_POINT_ESTIMATE_STATUS,
            SAMPLED_SUMMARY_AUTHORITY_SCOPE,
            V1SampledSummaryAcquisitionSource,
        )

        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError(
                "sampled diagnostic acquisition requires an exact campaign binding"
            )
        if type(self.current_mask) is not HardMask:
            raise TypeError(
                "sampled diagnostic acquisition requires an exact current mask"
            )
        if type(self.sampled_source) is not V1SampledSummaryAcquisitionSource:
            raise TypeError(
                "sampled diagnostic acquisition requires its exact summary source"
            )
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, "generation_index"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, "route_count"),
        )
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "context_population_hash",
            "sampling_law_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "raw_logit_stream_source_hash",
            "raw_logit_coordinate",
            "sampled_summary_source_hash",
            "discovery_source_receipt_hash",
            "discovery_admission_hash",
            "geometry_receipt_hash",
            "geometry_result_hash",
            "geometry_state_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "source_measurement_axis_hash",
            "corrected_moments_state_hash",
            "signed_point_hash",
            "absolute_point_hash",
            "corrected_point_estimate_hash",
            "corrected_point_status",
            "response_exactification_replay_receipt_hash",
            "score_order_registration_hash",
            "score_order_look_receipt_hash",
            "score_order_confidence_box_hash",
            "score_order_regret_receipt_axis_hash",
            "permutation_lineage_hash",
            "validation_firewall_hash",
            "terminal_validation_population_hash",
            "terminal_validation_policy_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))

        source = self.sampled_source
        discovery = source.discovery
        discovery_receipt = discovery.receipt
        geometry_receipt = source.receipt
        geometry_result = source.geometry_result
        geometry = geometry_result.geometry
        replay = discovery.score_order_replay
        replay_hash = replay.assert_integrity()
        moments_hash = discovery.moments.assert_integrity()
        fd_snapshot_hash = discovery.fd_snapshot.assert_integrity()
        fd_design_hash = discovery.fd_design.assert_integrity()
        discovery.axis_bridge.assert_integrity()
        ordered_axis_hash = ordered_string_axis_hash(
            geometry.route_ids,
            name="sampled diagnostic geometry route axis",
        )
        contract = self.binding.contract
        exact_coordinates = (
            (self.campaign_contract_hash, contract.campaign_contract_hash),
            (self.campaign_binding_hash, self.binding.binding_hash),
            (self.registration_hash, self.binding.registration.registration_hash),
            (self.model_identity_hash, discovery_receipt.model_identity_hash),
            (self.query_contract_hash, discovery_receipt.query_contract_hash),
            (self.preprocessing_hash, discovery_receipt.preprocessing_hash),
            (self.context_population_hash, contract.context_population_hash),
            (self.sampling_law_hash, contract.sampling_law_hash),
            (self.current_mask_hash, self.current_mask.mask_hash),
            (self.physical_route_axis_hash, self.current_mask.route_axis_hash),
            (self.ordered_route_axis_hash, discovery_receipt.ordered_route_axis_hash),
            (self.ordered_route_axis_hash, ordered_axis_hash),
            (self.route_count, self.current_mask.route_count),
            (self.route_count, len(geometry.route_ids)),
            (self.referent_id, contract.referent_id),
            (self.context_mixture_hash, discovery_receipt.context_mixture_hash),
            (self.calibration_identity, contract.calibration_identity),
            (self.calibration_identity, geometry.calibration_state_id),
            (
                self.raw_logit_stream_source_hash,
                discovery_receipt.raw_logit_stream_source_hash,
            ),
            (self.raw_logit_coordinate, discovery_receipt.raw_logit_coordinate),
            (self.generation_index, discovery_receipt.proposal_generation),
            (self.sampled_summary_source_hash, source.source_hash),
            (
                self.discovery_source_receipt_hash,
                discovery_receipt.receipt_hash,
            ),
            (self.discovery_admission_hash, discovery.admission_hash),
            (self.geometry_receipt_hash, geometry_receipt.receipt_hash),
            (self.geometry_result_hash, geometry_result.result_hash),
            (self.geometry_state_hash, geometry.content_hash),
            (self.fd_design_hash, fd_design_hash),
            (self.fd_snapshot_hash, fd_snapshot_hash),
            (
                self.source_measurement_axis_hash,
                discovery_receipt.source_measurement_axis_hash,
            ),
            (self.corrected_moments_state_hash, moments_hash),
            (self.signed_point_hash, discovery_receipt.signed_point_hash),
            (self.absolute_point_hash, discovery_receipt.absolute_point_hash),
            (
                self.corrected_point_estimate_hash,
                discovery_receipt.corrected_point_estimate_hash,
            ),
            (self.corrected_point_status, discovery_receipt.point_status),
            (self.response_exactification_replay_receipt_hash, replay_hash),
            (
                self.response_exactification_replay_receipt_hash,
                discovery_receipt.response_exactification_replay_receipt_hash,
            ),
            (
                self.score_order_registration_hash,
                replay.score_order_registration_hash,
            ),
            (self.score_order_look_receipt_hash, replay.score_order_look_receipt_hash),
            (self.score_order_confidence_box_hash, replay.confidence_box_hash),
            (
                self.score_order_regret_receipt_axis_hash,
                discovery_receipt.score_order_regret_receipt_axis_hash,
            ),
            (self.permutation_lineage_hash, discovery_receipt.permutation_lineage_hash),
            (self.validation_firewall_hash, discovery_receipt.validation_firewall_hash),
            (
                self.terminal_validation_population_hash,
                contract.terminal_validation_population_hash,
            ),
            (
                self.terminal_validation_policy_hash,
                contract.terminal_validation_policy_hash,
            ),
        )
        if any(left != right for left, right in exact_coordinates):
            raise ValueError(
                "sampled diagnostic acquisition metadata differs from its source graph"
            )
        if (
            discovery.binding.binding_hash != self.binding.binding_hash
            or discovery.current_mask.mask_hash != self.current_mask.mask_hash
            or geometry.measured_mask_hash != self.current_mask.mask_hash
            or self.physical_route_axis_hash == self.ordered_route_axis_hash
            or geometry.derivation_receipt is not None
            or self.corrected_point_status != SAMPLED_POINT_ESTIMATE_STATUS
        ):
            raise ValueError(
                "sampled diagnostic acquisition is stale or implies an exact stream"
            )
        if (
            discovery_receipt.authority_scope != SAMPLED_SUMMARY_AUTHORITY_SCOPE
            or geometry_receipt.authority_scope != SAMPLED_SUMMARY_AUTHORITY_SCOPE
            or discovery_receipt.census_authority is not False
            or geometry_receipt.census_authority is not False
            or discovery_receipt.raw_context_route_rows_retained is not False
            or geometry_receipt.raw_context_route_rows_retained is not False
            or discovery_receipt.full_per_draw_source_row_hash_ledger_retained
            is not False
            or geometry_receipt.full_per_draw_source_row_hash_ledger_retained
            is not False
            or discovery_receipt.source_row_proof_authority is not False
            or geometry_receipt.source_row_proof_authority is not False
            or discovery_receipt.fd_stream_acquisition_issued is not False
            or discovery_receipt.exact_tangent_price_acquisition_issued is not False
            or geometry_receipt.standard_fd_stream_acquisition_compatible is not False
            or geometry_receipt.exact_tangent_price_acquisition_compatible is not False
        ):
            raise PermissionError(
                "sampled diagnostic source claims unavailable exact evidence"
            )
        if (
            self.sampled_summary_compatible is not True
            or self.census_authority is not False
            or self.raw_context_route_rows_retained is not False
            or self.full_per_draw_source_row_hash_ledger_retained is not False
            or self.source_row_proof_authority is not False
            or self.exact_stream_rows_available is not False
            or self.standard_fd_stream_acquisition_compatible is not False
            or self.exact_tangent_price_acquisition_compatible is not False
            or self.exact_candidate_numeric_state_issued is not False
            or self.score_order_authority is not False
            or self.fq_certification_authority is not False
            or self.candidate_composition_authority is not False
            or self.candidate_freeze_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.validation_inputs_present is not False
            or self.proposal_only is not True
            or self.authority_scope != _SAMPLED_DIAGNOSTIC_AUTHORITY_SCOPE
            or self.schema != _SAMPLED_DIAGNOSTIC_ACQUISITION_SCHEMA
        ):
            raise PermissionError(
                "sampled diagnostic acquisition crossed its authority boundary"
            )
        metadata = dict(self.metadata)
        shadowed = sorted(
            _SAMPLED_DIAGNOSTIC_RESERVED_METADATA_KEYS.intersection(metadata)
        )
        if shadowed:
            raise ValueError(
                "sampled diagnostic metadata cannot shadow canonical fields: "
                f"{shadowed}"
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze(metadata, "sampled diagnostic metadata"),
        )

    @property
    def acquisition_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "campaign_contract_hash": self.campaign_contract_hash,
                "campaign_binding_hash": self.campaign_binding_hash,
                "registration_hash": self.registration_hash,
                "model_identity_hash": self.model_identity_hash,
                "query_contract_hash": self.query_contract_hash,
                "preprocessing_hash": self.preprocessing_hash,
                "generation_index": self.generation_index,
                "current_mask_hash": self.current_mask_hash,
                "physical_route_axis_hash": self.physical_route_axis_hash,
                "ordered_route_axis_hash": self.ordered_route_axis_hash,
                "route_count": self.route_count,
                "referent_id": self.referent_id,
                "context_population_hash": self.context_population_hash,
                "sampling_law_hash": self.sampling_law_hash,
                "context_mixture_hash": self.context_mixture_hash,
                "calibration_identity": self.calibration_identity,
                "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
                "raw_logit_coordinate": self.raw_logit_coordinate,
                "sampled_summary_source_hash": self.sampled_summary_source_hash,
                "discovery_source_receipt_hash": self.discovery_source_receipt_hash,
                "discovery_admission_hash": self.discovery_admission_hash,
                "geometry_receipt_hash": self.geometry_receipt_hash,
                "geometry_result_hash": self.geometry_result_hash,
                "geometry_state_hash": self.geometry_state_hash,
                "fd_design_hash": self.fd_design_hash,
                "fd_snapshot_hash": self.fd_snapshot_hash,
                "source_measurement_axis_hash": self.source_measurement_axis_hash,
                "corrected_moments_state_hash": self.corrected_moments_state_hash,
                "signed_point_hash": self.signed_point_hash,
                "absolute_point_hash": self.absolute_point_hash,
                "corrected_point_estimate_hash": self.corrected_point_estimate_hash,
                "corrected_point_status": self.corrected_point_status,
                "response_exactification_replay_receipt_hash": (
                    self.response_exactification_replay_receipt_hash
                ),
                "score_order_registration_hash": self.score_order_registration_hash,
                "score_order_look_receipt_hash": self.score_order_look_receipt_hash,
                "score_order_confidence_box_hash": (
                    self.score_order_confidence_box_hash
                ),
                "score_order_regret_receipt_axis_hash": (
                    self.score_order_regret_receipt_axis_hash
                ),
                "permutation_lineage_hash": self.permutation_lineage_hash,
                "validation_firewall_hash": self.validation_firewall_hash,
                "terminal_validation_population_hash": (
                    self.terminal_validation_population_hash
                ),
                "terminal_validation_policy_hash": (
                    self.terminal_validation_policy_hash
                ),
                "sampled_summary_compatible": self.sampled_summary_compatible,
                "census_authority": self.census_authority,
                "raw_context_route_rows_retained": (
                    self.raw_context_route_rows_retained
                ),
                "full_per_draw_source_row_hash_ledger_retained": (
                    self.full_per_draw_source_row_hash_ledger_retained
                ),
                "source_row_proof_authority": self.source_row_proof_authority,
                "exact_stream_rows_available": self.exact_stream_rows_available,
                "standard_fd_stream_acquisition_compatible": (
                    self.standard_fd_stream_acquisition_compatible
                ),
                "exact_tangent_price_acquisition_compatible": (
                    self.exact_tangent_price_acquisition_compatible
                ),
                "exact_candidate_numeric_state_issued": (
                    self.exact_candidate_numeric_state_issued
                ),
                "score_order_authority": self.score_order_authority,
                "fq_certification_authority": self.fq_certification_authority,
                "candidate_composition_authority": (
                    self.candidate_composition_authority
                ),
                "candidate_freeze_authority": self.candidate_freeze_authority,
                "bce_oracle_authority": self.bce_oracle_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "validation_inputs_present": self.validation_inputs_present,
                "authority_scope": self.authority_scope,
                "proposal_only": self.proposal_only,
                "metadata": self.metadata,
            },
            prefix="partition-free-v1-sampled-summary-diagnostic-acquisition-",
        )

    @property
    def sampled_signed_point(self) -> np.ndarray:
        """Read-only sampled signed point; never an exact tangent price."""

        return self.sampled_source.discovery.signed_point

    @property
    def sampled_absolute_point(self) -> np.ndarray:
        """Read-only sampled magnitude point; never a census value."""

        return self.sampled_source.discovery.absolute_point

    @property
    def sampled_geometry_state(self) -> GlobalTrustGeometryState:
        """Assembled sampled FD summary without a raw-stream derivation receipt."""

        return self.sampled_source.geometry_result.geometry

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def can_certify_decision(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError(
            "sampled diagnostic acquisition has no oracle or mask authority"
        )


@dataclass(frozen=True, slots=True)
class V1SampledCorrectedPointNumericLineage:
    """Hash-bearing lineage for later, explicitly F/Q-gated composition.

    This is not :class:`V1CandidateNumericState`.  It binds the sampled point
    and response geometry that a later composer may consume after a decisive
    selected-shell F/Q commit and an independently validated atom-cost state.
    """

    diagnostic_acquisition: V1SampledSummaryDiagnosticAcquisition = field(
        repr=False,
        compare=False,
    )
    diagnostic_acquisition_hash: str
    sampled_summary_source_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    generation_index: int
    parent_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    raw_logit_stream_source_hash: str
    raw_logit_coordinate: str
    corrected_moments_state_hash: str
    signed_point_hash: str
    absolute_point_hash: str
    corrected_point_estimate_hash: str
    corrected_point_status: str
    source_geometry_state_hash: str
    source_geometry_result_hash: str
    response_exactification_replay_receipt_hash: str
    score_order_registration_hash: str
    score_order_look_receipt_hash: str
    score_order_confidence_box_hash: str
    score_order_regret_receipt_axis_hash: str
    permutation_lineage_hash: str
    fq_composition_gate_policy_id: str = SAMPLED_FQ_COMPOSITION_GATE_POLICY_ID
    sampled_point_estimate: bool = True
    census_authority: bool = False
    exact_stream_rows_available: bool = False
    standard_fd_stream_acquisition_compatible: bool = False
    exact_tangent_price_acquisition_compatible: bool = False
    exact_candidate_numeric_state_issued: bool = False
    separate_atom_cost_state_required: bool = True
    decisive_fq_commit_required: bool = True
    score_order_authority: bool = False
    fq_certification_authority: bool = False
    candidate_composition_authority: bool = False
    candidate_freeze_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    validation_inputs_present: bool = False
    proposal_only: bool = True
    authority_scope: str = _SAMPLED_DIAGNOSTIC_AUTHORITY_SCOPE
    schema: str = _SAMPLED_CORRECTED_POINT_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.diagnostic_acquisition) is not (
            V1SampledSummaryDiagnosticAcquisition
        ):
            raise TypeError(
                "sampled numeric lineage requires its exact diagnostic acquisition"
            )
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, "generation_index"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, "route_count"),
        )
        for name in (
            "diagnostic_acquisition_hash",
            "sampled_summary_source_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
            "parent_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "raw_logit_stream_source_hash",
            "raw_logit_coordinate",
            "corrected_moments_state_hash",
            "signed_point_hash",
            "absolute_point_hash",
            "corrected_point_estimate_hash",
            "corrected_point_status",
            "source_geometry_state_hash",
            "source_geometry_result_hash",
            "response_exactification_replay_receipt_hash",
            "score_order_registration_hash",
            "score_order_look_receipt_hash",
            "score_order_confidence_box_hash",
            "score_order_regret_receipt_axis_hash",
            "permutation_lineage_hash",
            "fq_composition_gate_policy_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        acquisition = self.diagnostic_acquisition
        exact_coordinates = (
            (self.diagnostic_acquisition_hash, acquisition.acquisition_hash),
            (self.sampled_summary_source_hash, acquisition.sampled_summary_source_hash),
            (self.campaign_contract_hash, acquisition.campaign_contract_hash),
            (self.campaign_binding_hash, acquisition.campaign_binding_hash),
            (self.registration_hash, acquisition.registration_hash),
            (self.generation_index, acquisition.generation_index),
            (self.parent_mask_hash, acquisition.current_mask_hash),
            (self.physical_route_axis_hash, acquisition.physical_route_axis_hash),
            (self.ordered_route_axis_hash, acquisition.ordered_route_axis_hash),
            (self.route_count, acquisition.route_count),
            (self.referent_id, acquisition.referent_id),
            (self.context_mixture_hash, acquisition.context_mixture_hash),
            (self.calibration_identity, acquisition.calibration_identity),
            (
                self.raw_logit_stream_source_hash,
                acquisition.raw_logit_stream_source_hash,
            ),
            (self.raw_logit_coordinate, acquisition.raw_logit_coordinate),
            (
                self.corrected_moments_state_hash,
                acquisition.corrected_moments_state_hash,
            ),
            (self.signed_point_hash, acquisition.signed_point_hash),
            (self.absolute_point_hash, acquisition.absolute_point_hash),
            (
                self.corrected_point_estimate_hash,
                acquisition.corrected_point_estimate_hash,
            ),
            (self.corrected_point_status, acquisition.corrected_point_status),
            (self.source_geometry_state_hash, acquisition.geometry_state_hash),
            (self.source_geometry_result_hash, acquisition.geometry_result_hash),
            (
                self.response_exactification_replay_receipt_hash,
                acquisition.response_exactification_replay_receipt_hash,
            ),
            (
                self.score_order_registration_hash,
                acquisition.score_order_registration_hash,
            ),
            (
                self.score_order_look_receipt_hash,
                acquisition.score_order_look_receipt_hash,
            ),
            (
                self.score_order_confidence_box_hash,
                acquisition.score_order_confidence_box_hash,
            ),
            (
                self.score_order_regret_receipt_axis_hash,
                acquisition.score_order_regret_receipt_axis_hash,
            ),
            (self.permutation_lineage_hash, acquisition.permutation_lineage_hash),
        )
        if any(left != right for left, right in exact_coordinates):
            raise ValueError(
                "sampled corrected-point lineage differs from its acquisition"
            )
        if (
            self.fq_composition_gate_policy_id
            != SAMPLED_FQ_COMPOSITION_GATE_POLICY_ID
            or self.sampled_point_estimate is not True
            or self.census_authority is not False
            or self.exact_stream_rows_available is not False
            or self.standard_fd_stream_acquisition_compatible is not False
            or self.exact_tangent_price_acquisition_compatible is not False
            or self.exact_candidate_numeric_state_issued is not False
            or self.separate_atom_cost_state_required is not True
            or self.decisive_fq_commit_required is not True
            or self.score_order_authority is not False
            or self.fq_certification_authority is not False
            or self.candidate_composition_authority is not False
            or self.candidate_freeze_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.validation_inputs_present is not False
            or self.proposal_only is not True
            or self.authority_scope != _SAMPLED_DIAGNOSTIC_AUTHORITY_SCOPE
            or self.schema != _SAMPLED_CORRECTED_POINT_LINEAGE_SCHEMA
        ):
            raise PermissionError(
                "sampled corrected-point lineage crossed its authority boundary"
            )

    @property
    def lineage_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "diagnostic_acquisition_hash": self.diagnostic_acquisition_hash,
                "sampled_summary_source_hash": self.sampled_summary_source_hash,
                "campaign_contract_hash": self.campaign_contract_hash,
                "campaign_binding_hash": self.campaign_binding_hash,
                "registration_hash": self.registration_hash,
                "generation_index": self.generation_index,
                "parent_mask_hash": self.parent_mask_hash,
                "physical_route_axis_hash": self.physical_route_axis_hash,
                "ordered_route_axis_hash": self.ordered_route_axis_hash,
                "route_count": self.route_count,
                "referent_id": self.referent_id,
                "context_mixture_hash": self.context_mixture_hash,
                "calibration_identity": self.calibration_identity,
                "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
                "raw_logit_coordinate": self.raw_logit_coordinate,
                "corrected_moments_state_hash": self.corrected_moments_state_hash,
                "signed_point_hash": self.signed_point_hash,
                "absolute_point_hash": self.absolute_point_hash,
                "corrected_point_estimate_hash": self.corrected_point_estimate_hash,
                "corrected_point_status": self.corrected_point_status,
                "source_geometry_state_hash": self.source_geometry_state_hash,
                "source_geometry_result_hash": self.source_geometry_result_hash,
                "response_exactification_replay_receipt_hash": (
                    self.response_exactification_replay_receipt_hash
                ),
                "score_order_registration_hash": self.score_order_registration_hash,
                "score_order_look_receipt_hash": self.score_order_look_receipt_hash,
                "score_order_confidence_box_hash": (
                    self.score_order_confidence_box_hash
                ),
                "score_order_regret_receipt_axis_hash": (
                    self.score_order_regret_receipt_axis_hash
                ),
                "permutation_lineage_hash": self.permutation_lineage_hash,
                "fq_composition_gate_policy_id": self.fq_composition_gate_policy_id,
                "sampled_point_estimate": self.sampled_point_estimate,
                "census_authority": self.census_authority,
                "exact_stream_rows_available": self.exact_stream_rows_available,
                "standard_fd_stream_acquisition_compatible": (
                    self.standard_fd_stream_acquisition_compatible
                ),
                "exact_tangent_price_acquisition_compatible": (
                    self.exact_tangent_price_acquisition_compatible
                ),
                "exact_candidate_numeric_state_issued": (
                    self.exact_candidate_numeric_state_issued
                ),
                "separate_atom_cost_state_required": (
                    self.separate_atom_cost_state_required
                ),
                "decisive_fq_commit_required": self.decisive_fq_commit_required,
                "score_order_authority": self.score_order_authority,
                "fq_certification_authority": self.fq_certification_authority,
                "candidate_composition_authority": (
                    self.candidate_composition_authority
                ),
                "candidate_freeze_authority": self.candidate_freeze_authority,
                "bce_oracle_authority": self.bce_oracle_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "validation_inputs_present": self.validation_inputs_present,
                "authority_scope": self.authority_scope,
                "proposal_only": self.proposal_only,
            },
            prefix="partition-free-v1-sampled-corrected-point-numeric-lineage-",
        )

    @property
    def sampled_signed_point(self) -> np.ndarray:
        return self.diagnostic_acquisition.sampled_signed_point

    @property
    def sampled_absolute_point(self) -> np.ndarray:
        return self.diagnostic_acquisition.sampled_absolute_point

    @property
    def sampled_geometry_state(self) -> GlobalTrustGeometryState:
        return self.diagnostic_acquisition.sampled_geometry_state

    @property
    def parent_mask(self) -> HardMask:
        return self.diagnostic_acquisition.current_mask

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def claim_execution_certified(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError(
            "sampled corrected-point lineage requires F/Q and has no decision authority"
        )


def issue_v1_sampled_summary_diagnostic_acquisition(
    *,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    sampled_source: "V1SampledSummaryAcquisitionSource",
    metadata: Mapping[str, Any] | None = None,
) -> V1SampledSummaryDiagnosticAcquisition:
    """Issue one replay-bound sampled diagnostic without exact coercion."""

    from .modern_v1_sampled_acquisition_adapters import (
        V1SampledSummaryAcquisitionSource,
    )

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError(
            "sampled diagnostic issuance requires an exact campaign binding"
        )
    if type(current_mask) is not HardMask:
        raise TypeError("sampled diagnostic issuance requires an exact current mask")
    if type(sampled_source) is not V1SampledSummaryAcquisitionSource:
        raise TypeError("sampled diagnostic issuance requires its exact summary source")
    generation = _nonnegative_int(generation_index, "generation_index")
    source = sampled_source.discovery.receipt
    replay = sampled_source.discovery.score_order_replay
    geometry = sampled_source.geometry_result.geometry
    return V1SampledSummaryDiagnosticAcquisition(
        binding=binding,
        current_mask=current_mask,
        sampled_source=sampled_source,
        generation_index=generation,
        campaign_contract_hash=binding.contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        registration_hash=binding.registration.registration_hash,
        model_identity_hash=source.model_identity_hash,
        query_contract_hash=source.query_contract_hash,
        preprocessing_hash=source.preprocessing_hash,
        context_population_hash=source.context_population_hash,
        sampling_law_hash=source.sampling_law_hash,
        current_mask_hash=current_mask.mask_hash,
        physical_route_axis_hash=current_mask.route_axis_hash,
        ordered_route_axis_hash=source.ordered_route_axis_hash,
        route_count=current_mask.route_count,
        referent_id=binding.contract.referent_id,
        context_mixture_hash=source.context_mixture_hash,
        calibration_identity=source.calibration_identity,
        raw_logit_stream_source_hash=source.raw_logit_stream_source_hash,
        raw_logit_coordinate=source.raw_logit_coordinate,
        sampled_summary_source_hash=sampled_source.source_hash,
        discovery_source_receipt_hash=source.receipt_hash,
        discovery_admission_hash=sampled_source.discovery.admission_hash,
        geometry_receipt_hash=sampled_source.receipt.receipt_hash,
        geometry_result_hash=sampled_source.geometry_result.result_hash,
        geometry_state_hash=geometry.content_hash,
        fd_design_hash=source.fd_design_hash,
        fd_snapshot_hash=source.fd_snapshot_hash,
        source_measurement_axis_hash=source.source_measurement_axis_hash,
        corrected_moments_state_hash=source.corrected_moments_state_hash,
        signed_point_hash=source.signed_point_hash,
        absolute_point_hash=source.absolute_point_hash,
        corrected_point_estimate_hash=source.corrected_point_estimate_hash,
        corrected_point_status=source.point_status,
        response_exactification_replay_receipt_hash=(
            replay.assert_integrity()
        ),
        score_order_registration_hash=replay.score_order_registration_hash,
        score_order_look_receipt_hash=replay.score_order_look_receipt_hash,
        score_order_confidence_box_hash=replay.confidence_box_hash,
        score_order_regret_receipt_axis_hash=(
            source.score_order_regret_receipt_axis_hash
        ),
        permutation_lineage_hash=source.permutation_lineage_hash,
        validation_firewall_hash=source.validation_firewall_hash,
        terminal_validation_population_hash=(
            source.terminal_validation_population_hash
        ),
        terminal_validation_policy_hash=source.terminal_validation_policy_hash,
        metadata={} if metadata is None else metadata,
    )


def issue_v1_sampled_corrected_point_numeric_lineage(
    acquisition: V1SampledSummaryDiagnosticAcquisition,
) -> V1SampledCorrectedPointNumericLineage:
    """Issue diagnostic numeric lineage for a future F/Q-gated composer."""

    if type(acquisition) is not V1SampledSummaryDiagnosticAcquisition:
        raise TypeError(
            "sampled corrected-point issuance requires its exact diagnostic acquisition"
        )
    return V1SampledCorrectedPointNumericLineage(
        diagnostic_acquisition=acquisition,
        diagnostic_acquisition_hash=acquisition.acquisition_hash,
        sampled_summary_source_hash=acquisition.sampled_summary_source_hash,
        campaign_contract_hash=acquisition.campaign_contract_hash,
        campaign_binding_hash=acquisition.campaign_binding_hash,
        registration_hash=acquisition.registration_hash,
        generation_index=acquisition.generation_index,
        parent_mask_hash=acquisition.current_mask_hash,
        physical_route_axis_hash=acquisition.physical_route_axis_hash,
        ordered_route_axis_hash=acquisition.ordered_route_axis_hash,
        route_count=acquisition.route_count,
        referent_id=acquisition.referent_id,
        context_mixture_hash=acquisition.context_mixture_hash,
        calibration_identity=acquisition.calibration_identity,
        raw_logit_stream_source_hash=acquisition.raw_logit_stream_source_hash,
        raw_logit_coordinate=acquisition.raw_logit_coordinate,
        corrected_moments_state_hash=acquisition.corrected_moments_state_hash,
        signed_point_hash=acquisition.signed_point_hash,
        absolute_point_hash=acquisition.absolute_point_hash,
        corrected_point_estimate_hash=acquisition.corrected_point_estimate_hash,
        corrected_point_status=acquisition.corrected_point_status,
        source_geometry_state_hash=acquisition.geometry_state_hash,
        source_geometry_result_hash=acquisition.geometry_result_hash,
        response_exactification_replay_receipt_hash=(
            acquisition.response_exactification_replay_receipt_hash
        ),
        score_order_registration_hash=acquisition.score_order_registration_hash,
        score_order_look_receipt_hash=acquisition.score_order_look_receipt_hash,
        score_order_confidence_box_hash=acquisition.score_order_confidence_box_hash,
        score_order_regret_receipt_axis_hash=(
            acquisition.score_order_regret_receipt_axis_hash
        ),
        permutation_lineage_hash=acquisition.permutation_lineage_hash,
    )


def build_v1_fd_geometry_from_stream(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    route_ids: Sequence[str],
    batches: Sequence[V1FDStreamBatch],
    trust_spec: GlobalTrustGeometrySpec,
    calibration_identity: str,
    backend: FDBackend = "canonical_cpu",
    metadata: Mapping[str, Any] | None = None,
) -> V1FDStreamAcquisition:
    """Accumulate the exact logical row stream into canonical rank-8 FD."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("FD acquisition requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("FD acquisition requires an exact current hard mask")
    if type(trust_spec) is not GlobalTrustGeometrySpec:
        raise TypeError("FD acquisition requires an exact trust specification")
    routes = _aligned_strings(route_ids, "FD route_ids", unique=True)
    route_hash = ordered_string_axis_hash(routes, name="modern-v1 production FD route axis")
    contract = binding.contract
    calibration = _nonempty(calibration_identity, "calibration_identity")
    if (
        current_mask.route_axis_hash != contract.route_axis_hash
        or current_mask.route_count != len(routes)
    ):
        raise ValueError("FD current mask and campaign physical route axis differ")
    if calibration != contract.calibration_identity:
        raise ValueError("FD calibration differs from the campaign contract")
    if len(routes) < OPTION_A_FD_RANK:
        raise ValueError("production rank-8 FD requires at least eight routes")
    if trust_spec.initial_rank != OPTION_A_FD_RANK or trust_spec.maximum_rank < OPTION_A_FD_RANK:
        raise ValueError("modern-v1 production trust specification must start at rank 8")
    if backend != "canonical_cpu":
        raise ValueError("source-bound local acquisition supports canonical_cpu only")
    exact_batches = tuple(batches)
    if not exact_batches or any(type(batch) is not V1FDStreamBatch for batch in exact_batches):
        raise TypeError("FD acquisition requires exact V1FDStreamBatch values")

    seen_draws: set[str] = set()
    all_draws: list[str] = []
    all_sources: list[str] = []
    all_row_hashes: list[str] = []
    all_weights: list[float] = []
    measurement_hashes: list[str] = []
    response_count: int | None = None
    accumulator = FrequentDirectionsTrustAccumulator(
        route_ids=routes,
        sketch_rank=OPTION_A_FD_RANK,
    )
    for batch in exact_batches:
        if (
            batch.current_mask_hash != current_mask.mask_hash
            or batch.route_axis_hash != route_hash
            or batch.calibration_identity != calibration
            or batch.row_shape[2] != len(routes)
        ):
            raise ValueError(
                "FD batch identity or ordered route axis differs from the current acquisition"
            )
        if response_count is None:
            response_count = batch.row_shape[1]
        elif response_count != batch.row_shape[1]:
            raise ValueError("FD response axis changed across streamed batches")
        overlap = seen_draws.intersection(batch.draw_ids)
        if overlap:
            raise ValueError("FD logical draw IDs were reused across batches")
        seen_draws.update(batch.draw_ids)
        accumulator.update(
            batch.rows,
            context_ids=batch.draw_ids,
            weights=batch.weights,
        )
        all_draws.extend(batch.draw_ids)
        all_sources.extend(batch.source_context_ids)
        all_row_hashes.extend(global_trust_source_row_hash(row) for row in batch.rows)
        all_weights.extend(float(value) for value in batch.weights)
        measurement_hashes.append(batch.source_measurement_hash)

    if len(set(measurement_hashes)) != len(measurement_hashes):
        raise ValueError("FD source measurement hashes must be unique per batch")
    state = accumulator.finalize(
        spec=trust_spec,
        measured_mask_hash=current_mask.mask_hash,
        calibration_state_id=calibration,
        source_measurement_hashes=tuple(measurement_hashes),
        metadata={
            "modern_v1_campaign_contract_hash": contract.campaign_contract_hash,
            "modern_v1_campaign_binding_hash": binding.binding_hash,
            "logical_draw_count": len(all_draws),
            "physical_source_count": len(set(all_sources)),
            "response_count": response_count,
            "backend": backend,
            **dict(metadata or {}),
        },
    )
    ledger = V1FDSourceLedger(
        draw_ids=tuple(all_draws),
        source_context_ids=tuple(all_sources),
        source_row_hashes=tuple(all_row_hashes),
        source_measurement_hashes=tuple(measurement_hashes),
        pre_outcome_stream_weight_hash=global_trust_stream_weight_hash(all_weights),
        expected_ordered_route_axis_hash=route_hash,
        expected_measured_mask_hash=current_mask.mask_hash,
        expected_calibration_state_id=calibration,
    )
    adapter = adapt_global_trust_geometry_to_v1_lineage(
        state,
        source_ledger=ledger,
    )
    return V1FDStreamAcquisition(
        campaign_contract_hash=contract.campaign_contract_hash,
        current_mask_hash=current_mask.mask_hash,
        route_axis_hash=route_hash,
        backend=backend,
        batches=exact_batches,
        geometry_state=state,
        source_ledger=ledger,
        adapter_result=adapter,
    )


def admit_v1_fd_geometry_from_producer(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    route_ids: Sequence[str],
    geometry_state: GlobalTrustGeometryState,
    source_ledger: V1FDSourceLedger,
    backend: FDBackend = "torch_cuda",
) -> V1FDStreamAcquisition:
    """Admit one producer-owned exact FD state without retaining raw host rows.

    This is the production entry point.  The producer must already have
    emitted the immutable state and accumulator derivation receipt from the
    exact ordered stream.  The modern-v1 adapter independently validates that
    receipt against the exact draw/source/hash ledger before this function
    issues a bounded proposal-source admission.
    """

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("FD producer admission requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("FD producer admission requires an exact current hard mask")
    if type(geometry_state) is not GlobalTrustGeometryState:
        raise TypeError("FD producer admission requires an exact trust geometry state")
    if type(source_ledger) is not V1FDSourceLedger:
        raise TypeError("FD producer admission requires an exact source ledger")
    if backend != "torch_cuda":
        raise ValueError("producer-state admission currently requires torch_cuda")

    routes = _aligned_strings(route_ids, "FD producer route_ids", unique=True)
    route_hash = ordered_string_axis_hash(
        routes,
        name="modern-v1 production FD route axis",
    )
    contract = binding.contract
    geometry_route_hash = ordered_string_axis_hash(
        geometry_state.route_ids,
        name="modern-v1 producer geometry route axis",
    )
    if (
        current_mask.route_axis_hash != contract.route_axis_hash
        or current_mask.route_count != len(routes)
        or len(geometry_state.route_ids) != len(routes)
        or geometry_route_hash != route_hash
        or source_ledger.expected_ordered_route_axis_hash != route_hash
    ):
        raise ValueError(
            "FD producer state, physical mask, or ordered route-ID axis differs"
        )
    if len(routes) < OPTION_A_FD_RANK:
        raise ValueError("production rank-8 FD requires at least eight routes")
    if (
        geometry_state.spec.initial_rank != OPTION_A_FD_RANK
        or geometry_state.spec.maximum_rank < OPTION_A_FD_RANK
    ):
        raise ValueError("modern-v1 producer trust specification must start at rank 8")
    if (
        geometry_state.measured_mask_hash != current_mask.mask_hash
        or source_ledger.expected_measured_mask_hash != current_mask.mask_hash
        or geometry_state.calibration_state_id != contract.calibration_identity
        or source_ledger.expected_calibration_state_id != contract.calibration_identity
    ):
        raise ValueError("FD producer state is stale or uses another calibration")

    metadata = geometry_state.metadata
    if (
        metadata.get("modern_v1_campaign_contract_hash")
        != contract.campaign_contract_hash
        or metadata.get("modern_v1_campaign_binding_hash") != binding.binding_hash
        or metadata.get("backend") != "torch_cuda"
    ):
        raise ValueError("FD producer metadata is not bound to the exact campaign")
    if metadata.get("derivation_proof_mode") != "bounded_ordered_row_stream_exact_v1":
        raise ValueError("production CUDA admission requires the bounded exact source proof")
    if metadata.get("derivation_proof_full_tensor_host_snapshot") is not False:
        raise ValueError("production CUDA admission forbids a full host tensor snapshot")
    if metadata.get("derivation_proof_streaming_window_contexts") != 1:
        raise ValueError("production CUDA source proof must use a one-row window")

    adapter = adapt_global_trust_geometry_to_v1_lineage(
        geometry_state,
        source_ledger=source_ledger,
    )
    derivation = geometry_state.derivation_receipt
    if derivation is None:
        raise ValueError("CUDA producer state lacks its accumulator derivation receipt")
    expected_lineage_implementation = _CUDA_FD_LINEAGE_IMPLEMENTATION_BY_ALGORITHM.get(
        derivation.producer_algorithm_id
    )
    if (
        expected_lineage_implementation is None
        or adapter.lineage.implementation != expected_lineage_implementation
    ):
        raise ValueError(
            "producer state did not verify as its exact CUDA FD implementation"
        )
    peak_bytes = metadata.get("derivation_proof_peak_host_row_bytes")
    total_bytes = metadata.get("derivation_proof_total_host_transfer_bytes")
    receipt = V1FDProducerAdmissionReceipt(
        campaign_contract_hash=contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        current_mask_hash=current_mask.mask_hash,
        route_axis_hash=route_hash,
        backend=backend,
        source_state_hash=geometry_state.content_hash,
        source_ledger_hash=source_ledger.ledger_hash,
        accumulator_derivation_receipt_hash=derivation.receipt_hash,
        adapter_result_hash=adapter.result_hash,
        derivation_proof_mode=str(metadata["derivation_proof_mode"]),
        proof_streaming_window_contexts=metadata[
            "derivation_proof_streaming_window_contexts"
        ],
        proof_peak_host_row_bytes=peak_bytes,
        proof_total_host_transfer_bytes=total_bytes,
    )
    return V1FDStreamAcquisition(
        campaign_contract_hash=contract.campaign_contract_hash,
        current_mask_hash=current_mask.mask_hash,
        route_axis_hash=route_hash,
        backend=backend,
        batches=(),
        geometry_state=geometry_state,
        source_ledger=source_ledger,
        adapter_result=adapter,
        producer_admission_receipt=receipt,
    )


def fit_v1_current_mask_correction(
    *,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    feature_map: V1ActionFeatureMap,
    tangent_prices: V1TangentPriceState,
    atom_costs: V1AtomCostState,
    design: V1FiniteInterventionDesign,
    specification: V1InterventionalCorrectionSpec,
    outcomes: Sequence[MeasurementOutcome],
) -> V1CorrectionAcquisition:
    """Fit and independently verify the identified current-mask correction."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("correction acquisition requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("correction acquisition requires an exact current hard mask")
    physical_route_hash = binding.contract.route_axis_hash
    route_hash = feature_map.route_axis_hash
    if current_mask.route_axis_hash != physical_route_hash:
        raise ValueError(
            "correction current mask differs from the campaign physical route axis"
        )
    if tangent_prices.referent_id != binding.contract.referent_id:
        raise ValueError("correction tangent referent differs from the campaign")
    if tangent_prices.calibration_identity != binding.contract.calibration_identity:
        raise ValueError("correction tangent calibration differs from the campaign")
    if design.referent_id != binding.contract.referent_id:
        raise ValueError("correction design referent differs from the campaign")
    if design.calibration_identity != binding.contract.calibration_identity:
        raise ValueError("correction design calibration differs from the campaign")
    exact_outcomes = tuple(outcomes)
    result = fit_identified_interventional_correction(
        feature_map=feature_map,
        tangent_prices=tangent_prices,
        atom_costs=atom_costs,
        design=design,
        spec=specification,
        outcomes=exact_outcomes,
    )
    return V1CorrectionAcquisition(
        campaign_contract_hash=binding.contract.campaign_contract_hash,
        current_mask_hash=current_mask.mask_hash,
        route_axis_hash=route_hash,
        feature_map=feature_map,
        tangent_prices=tangent_prices,
        atom_costs=atom_costs,
        design=design,
        specification=specification,
        outcomes=exact_outcomes,
        result=result,
    )


def compose_v1_relinearized_acquisition(
    *,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    fd: V1FDStreamAcquisition,
    correction: (
        V1CorrectionAcquisition
        | V1AnalyticExactGateCorrectionAcquisition
    ),
    thresholds: V1ChartThresholds,
    domain: V1FiniteRemovalDomain,
    constraints: V1CandidateCompositionConstraints,
    context_mixture_hash: str,
    residual_diagonal_status: ResidualDiagonalStatus = "proposal_only",
    residual_bound_proof_hash: str | None = None,
    mixed_difference_envelope: V1MixedDifferenceEnvelope | None = None,
    refresh_reason: str = "fresh_current_mask_atomic_relinearization",
    logical_sequence: int = 0,
    candidate_composition_mode: Literal[
        "legacy_damage_rank",
        "frozen_resource_descending",
    ] = "legacy_damage_rank",
    metadata: Mapping[str, Any] | None = None,
) -> V1RelinearizedAcquisition:
    """Compose one exact deterministic partition-free queue."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("acquisition composition requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("acquisition composition requires an exact current hard mask")
    for value, expected, label in (
        (fd, V1FDStreamAcquisition, "FD acquisition"),
        (thresholds, V1ChartThresholds, "chart thresholds"),
        (domain, V1FiniteRemovalDomain, "finite removal domain"),
        (constraints, V1CandidateCompositionConstraints, "candidate constraints"),
    ):
        if type(value) is not expected:
            raise TypeError(f"acquisition composition requires exact {label}")
    if type(correction) not in {
        V1CorrectionAcquisition,
        V1AnalyticExactGateCorrectionAcquisition,
    }:
        raise TypeError("acquisition composition requires exact correction acquisition")
    generation = _nonnegative_int(generation_index, "generation_index")
    if candidate_composition_mode not in {
        "legacy_damage_rank",
        "frozen_resource_descending",
    }:
        raise ValueError("unsupported candidate composition mode")
    contract = binding.contract
    registration = binding.registration
    context_mixture = _nonempty(context_mixture_hash, "context_mixture_hash")
    lineage = tuple(_nonempty(value, "mask lineage hash") for value in committed_mask_lineage_hashes)
    transitions = tuple(
        _nonempty(value, "mask transition receipt hash")
        for value in committed_mask_transition_receipt_hashes
    )
    if (
        not lineage
        or lineage[-1] != current_mask.mask_hash
        or len(transitions) != len(lineage) - 1
        or len(set(lineage)) != len(lineage)
        or len(set(transitions)) != len(transitions)
    ):
        raise ValueError("acquisition requires complete unique exact mask ancestry")
    if (
        current_mask.route_axis_hash != contract.route_axis_hash
        or fd.current_mask_hash != current_mask.mask_hash
        or correction.current_mask_hash != current_mask.mask_hash
        or fd.campaign_contract_hash != contract.campaign_contract_hash
        or correction.campaign_contract_hash != contract.campaign_contract_hash
    ):
        raise ValueError("acquisition inputs are stale or belong to another campaign")
    if (
        fd.route_axis_hash != correction.route_axis_hash
        or domain.route_axis_hash != contract.route_axis_hash
        or domain.route_count != current_mask.route_count
    ):
        raise ValueError(
            "acquisition physical domain or numeric route-ID sources name another axis"
        )
    if domain.action_count > contract.maximum_candidate_count:
        raise ValueError("finite candidate domain exceeds the campaign candidate cap")
    if constraints.maximum_removed_count > contract.maximum_removed_count:
        raise ValueError("candidate constraints exceed the campaign removal cap")
    if domain.maximum_action_size > contract.maximum_removed_count:
        raise ValueError("finite candidate domain contains an over-cap removal action")
    correction_context_mixture = (
        correction.design.context_mixture_hash
        if type(correction) is V1CorrectionAcquisition
        else correction.context_mixture_hash
    )
    if correction_context_mixture != context_mixture:
        raise ValueError("correction and acquisition context mixtures differ")
    if correction.result.lineage.context_mixture_hash != context_mixture:
        raise ValueError("correction lineage and acquisition context mixture differ")

    current_live_flags = current_mask.to_bool_array()
    live_route_count = current_mask.retained_count
    if live_route_count <= 0:
        raise ValueError("cannot acquire a queue from an empty live route axis")
    live_hash = (
        fd.route_axis_hash
        if live_route_count == current_mask.route_count
        else _selected_route_id_axis_hash(
            fd.geometry_state.route_ids,
            current_live_flags,
        )
    )
    cumulative_removed_count = current_mask.route_count - current_mask.retained_count
    cumulative_payload: dict[str, object] = {
        "initial_mask_hash": lineage[0],
        "current_mask_hash": current_mask.mask_hash,
        "transition_receipt_hashes": transitions,
    }
    if cumulative_removed_count <= 1024:
        # Preserve historical identities for existing small-action campaigns.
        cumulative_payload["removed_route_indices"] = tuple(
            int(index) for index in np.flatnonzero(~current_live_flags)
        )
    else:
        # The exact complete mask already content-addresses every removed bit;
        # repeating millions of indices in this cumulative receipt is both
        # redundant and operationally prohibitive.
        cumulative_payload.update(
            {
                "schema": "partition_free_v1_cumulative_removal_v2",
                "removed_route_count": cumulative_removed_count,
                "removed_route_set_identity": content_hash(
                    {
                        "route_axis_hash": current_mask.route_axis_hash,
                        "current_mask_hash": current_mask.mask_hash,
                        "removed_route_count": cumulative_removed_count,
                    },
                    prefix="partition-free-v1-compact-cumulative-removal-set-",
                ),
            }
        )
    cumulative_hash = content_hash(
        cumulative_payload,
        prefix="partition-free-v1-cumulative-removal-",
    )
    exact_diagonal_hash = fd.adapter_result.lineage.metadata.get(
        "exact_residual_diagonal_hash"
    )
    if residual_diagonal_status == "absent":
        diagonal_receipt_hash = None
    else:
        if type(exact_diagonal_hash) is not str or not exact_diagonal_hash:
            raise ValueError("FD lineage lacks an exact residual-diagonal receipt")
        diagonal_receipt_hash = exact_diagonal_hash
    if residual_diagonal_status in {
        "global_loewner_bound",
        "action_class_quadratic_bound",
    } and residual_bound_proof_hash is None:
        raise ValueError("a hard residual bound status requires an external proof hash")
    if residual_diagonal_status == "action_class_quadratic_bound" and mixed_difference_envelope is None:
        raise ValueError("action-class residual status requires its typed envelope")

    axis_identity_metadata = (
        {}
        if fd.route_axis_hash == contract.route_axis_hash
        else {
            "physical_route_axis_hash": contract.route_axis_hash,
            "route_id_axis_hash": fd.route_axis_hash,
        }
    )
    chart_contract = V1ProposalChartBuildContract(
        registration_hash=registration.registration_hash,
        base_mask_hash=current_mask.mask_hash,
        current_mask_hash=current_mask.mask_hash,
        cumulative_removal_hash=cumulative_hash,
        # Each accepted change invalidates the preceding local chart.  A fresh
        # chart is therefore rooted at the exact current mask and has no local
        # chart transition history.  Complete run ancestry remains bound by
        # ``cumulative_hash`` and the enclosing acquisition identity.
        committed_mask_lineage_hashes=(current_mask.mask_hash,),
        committed_mask_transition_receipt_hashes=(),
        route_axis_hash=fd.route_axis_hash,
        live_route_axis_hash=live_hash,
        referent_id=contract.referent_id,
        context_mixture_hash=context_mixture,
        calibration_identity=contract.calibration_identity,
        thresholds=thresholds,
        residual_diagonal_status=residual_diagonal_status,
        residual_diagonal_receipt_hash=diagonal_receipt_hash,
        residual_bound_proof_hash=residual_bound_proof_hash,
        mixed_difference_envelope=mixed_difference_envelope,
        metadata={
            "campaign_contract_hash": contract.campaign_contract_hash,
            "campaign_binding_hash": binding.binding_hash,
            "generation_index": generation,
            "refresh_policy_id": contract.refresh_policy_id,
            "candidate_domain_policy_hash": contract.candidate_domain_policy_hash,
            "resource_cost_policy_hash": contract.resource_cost_policy_hash,
            "proposal_only": True,
            **axis_identity_metadata,
            **dict(metadata or {}),
        },
    )
    chart = build_v1_proposal_chart(
        chart_contract,
        fd_lineage=fd.adapter_result,
        correction_lineage=correction.result,
    )
    numeric_state = issue_v1_candidate_numeric_state(
        chart=chart,
        global_geometry_state=fd.geometry_state,
        fd_result=fd.adapter_result,
        correction_result=correction.result,
        atom_costs=correction.atom_costs,
        current_mask=current_mask,
    )
    composition = (
        compose_v1_candidates_exact(
            numeric_state=numeric_state,
            domain=domain,
            constraints=constraints,
        )
        if candidate_composition_mode == "legacy_damage_rank"
        else compose_v1_candidates_in_frozen_resource_descending_order(
            numeric_state=numeric_state,
            domain=domain,
            constraints=constraints,
        )
    )
    if (
        candidate_composition_mode == "frozen_resource_descending"
        and composition.composer_id
        != FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID
    ):
        raise RuntimeError("frozen frontier composer identity drifted")
    core_metadata = {
        "refresh_reason": refresh_reason,
        "logical_sequence": logical_sequence,
        "generation_index": generation,
        **axis_identity_metadata,
    }
    if candidate_composition_mode != "legacy_damage_rank":
        core_metadata["candidate_composition_mode"] = candidate_composition_mode
    acquisition_hash = content_hash(
        {
            "schema": _ACQUISITION_RESULT_SCHEMA,
            "campaign_binding_hash": binding.binding_hash,
            "generation_index": generation,
            "current_mask_hash": current_mask.mask_hash,
            "committed_mask_lineage_hashes": lineage,
            "committed_mask_transition_receipt_hashes": transitions,
            "fd_acquisition_hash": fd.acquisition_hash,
            "correction_acquisition_hash": correction.acquisition_hash,
            "chart_hash": chart.chart_hash,
            "numeric_state_hash": numeric_state.state_hash,
            "domain_hash": domain.domain_hash,
            "constraints_hash": constraints.constraints_hash,
            "composition_result_hash": composition.result_hash,
            "metadata": core_metadata,
        },
        prefix="partition-free-v1-relinearized-acquisition-",
    )
    receipt = V1AcquisitionGenerationReceipt(
        campaign_contract_hash=contract.campaign_contract_hash,
        campaign_binding_hash=binding.binding_hash,
        registration_hash=registration.registration_hash,
        generation_index=generation,
        current_mask_hash=current_mask.mask_hash,
        route_axis_hash=contract.route_axis_hash,
        acquisition_hash=acquisition_hash,
        geometry_state_hash=fd.geometry_state.content_hash,
        fd_adapter_result_hash=fd.adapter_result.result_hash,
        correction_result_hash=correction.result.result_receipt_hash,
        chart_hash=chart.chart_hash,
        numeric_state_hash=numeric_state.state_hash,
        finite_domain_hash=domain.domain_hash,
        constraints_hash=constraints.constraints_hash,
        composition_result_hash=composition.result_hash,
        ordered_candidate_ids=tuple(
            proposal.candidate_id for proposal in composition.ranked_proposals
        ),
        source_measurement_hashes=tuple(fd.geometry_state.source_measurement_hashes),
        refresh_reason=_nonempty(refresh_reason, "refresh_reason"),
        logical_sequence=_nonnegative_int(logical_sequence, "logical_sequence"),
        metadata={
            "context_mixture_hash": context_mixture,
            "residual_diagonal_status": residual_diagonal_status,
            "evaluated_action_count": composition.evaluated_action_count,
            "eligible_action_count": composition.eligible_action_count,
            "global_partition_free_order": True,
            "legacy_partition_identity_present": False,
            **axis_identity_metadata,
        },
    )
    result = V1RelinearizedAcquisition(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        fd=fd,
        correction=correction,
        chart=chart,
        numeric_state=numeric_state,
        domain=domain,
        constraints=constraints,
        composition=composition,
        receipt=receipt,
        metadata=core_metadata,
    )
    if result.acquisition_hash != acquisition_hash:
        raise RuntimeError("relinearized acquisition hash is not reproducible")
    return result


__all__ = [
    "FDBackend",
    "ResidualDiagonalStatus",
    "SAMPLED_FQ_COMPOSITION_GATE_POLICY_ID",
    "V1AnalyticExactGateCorrectionAcquisition",
    "V1CorrectionAcquisition",
    "V1FDProducerAdmissionReceipt",
    "V1FDStreamAcquisition",
    "V1FDStreamBatch",
    "V1FQAcquisitionGenerationReceipt",
    "V1FQAuthorizedAcquisition",
    "V1RelinearizedAcquisition",
    "V1SampledCorrectedPointNumericLineage",
    "V1SampledSummaryDiagnosticAcquisition",
    "admit_v1_fd_geometry_from_producer",
    "build_v1_fd_geometry_from_stream",
    "compose_v1_relinearized_acquisition",
    "fit_v1_current_mask_correction",
    "issue_v1_fq_authorized_acquisition",
    "issue_v1_sampled_corrected_point_numeric_lineage",
    "issue_v1_sampled_summary_diagnostic_acquisition",
]
