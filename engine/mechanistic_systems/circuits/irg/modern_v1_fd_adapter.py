"""Strict proposal-only adapter for modern-v1 Frequent-Directions lineage.

Historical trust-geometry states predate the modern-v1 source ledger and
remain parseable, but cannot enter this adapter.  This module performs one
fail-closed conversion from an exact accumulator-receipted
``GlobalTrustGeometryState`` snapshot to an inert lineage plus a typed
verification bundle.

In particular, the adapter derives every numerical claim from the state.  A
caller supplies identities and pre-outcome lineage only; it cannot supply a
factor hash, shrink value, or streamed norm.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from mechanistic_systems.core.axis_identity import ordered_string_axis_hash

from .contracts import GlobalTrustGeometrySpec, content_hash
from .modern_v1_core import (
    V1FDLineageVerificationReceipt,
    V1FrequentDirectionsLineage,
)
from .trust_geometry import (
    GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID,
    GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID,
    GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID,
    GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA,
    GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA,
    GlobalTrustGeometryDerivationReceipt,
    GlobalTrustGeometryState,
    global_trust_ordered_source_rows_hash,
)

CANONICAL_FD_ALGORITHM_ID: Final = GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID
CUDA_FD_ALGORITHM_ID: Final = GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID
CUDA_GRAM_FD_ALGORITHM_ID: Final = GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID
CUDA_FD_IMPLEMENTATION_ID: Final = "cuda_algebraically_equivalent"
CUDA_GRAM_FD_IMPLEMENTATION_ID: Final = "cuda_gram_algebraically_equivalent"
EFFECTIVE_FACTOR_DERIVATION_ID: Final = "global_trust_mean_plus_centered_factor_v1"
FD_SHRINK_DERIVATION_ID: Final = "global_trust_normalized_cumulative_fd_shrink_v1"
EXACT_STREAMED_NORM_DERIVATION_ID: Final = (
    "global_trust_exact_stream_frobenius_from_fd_remainder_diagonal_v1"
)

_SOURCE_LEDGER_SCHEMA: Final = "partition_free_v1_fd_source_ledger_v1"
_CUDA_EQUIVALENCE_SCHEMA: Final = "partition_free_v1_fd_cuda_equivalence_receipt_v1"
_ADAPTER_SCHEMA: Final = "partition_free_v1_global_trust_fd_adapter_v1"
_ADAPTER_RESULT_SCHEMA: Final = "partition_free_v1_fd_adapter_result_v1"
_FACTOR_HASH_PREFIX: Final = "partition-free-v1-effective-fd-factor-"
_FD_ADAPTER_RESULT_TOKEN = object()


def _strict_nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _strict_string_tuple(
    values: Sequence[str],
    label: str,
    *,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of strings")
    result = tuple(values)
    if not result or any(type(value) is not str or not value for value in result):
        raise ValueError(f"{label} must be a nonempty sequence of nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique values")
    return result


def _require_bool(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label} must be exactly {expected!r}")
    return expected


@dataclass(frozen=True, slots=True)
class V1FDSourceLedger:
    """Frozen draw/source ledger for one proposal-only FD stream.

    Draw IDs are the unique statistical axis.  Physical sources and their row
    hashes deliberately need not be unique because sampling with replacement
    is valid.  ``pre_outcome_stream_weight_hash`` records the already-frozen
    weighting law/realization identity; it is never reconstructed from an
    outcome-bearing state.
    """

    draw_ids: tuple[str, ...]
    source_context_ids: tuple[str, ...]
    source_row_hashes: tuple[str, ...]
    source_measurement_hashes: tuple[str, ...]
    pre_outcome_stream_weight_hash: str
    expected_ordered_route_axis_hash: str
    expected_measured_mask_hash: str
    expected_calibration_state_id: str
    proposal_only: bool = True
    claim_execution_certified: bool = False
    schema: str = _SOURCE_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        draws = _strict_string_tuple(self.draw_ids, "draw_ids", unique=True)
        sources = _strict_string_tuple(
            self.source_context_ids,
            "source_context_ids",
            unique=False,
        )
        rows = _strict_string_tuple(
            self.source_row_hashes,
            "source_row_hashes",
            unique=False,
        )
        if len(draws) != len(sources) or len(draws) != len(rows):
            raise ValueError(
                "draw_ids, source_context_ids, and source_row_hashes must align one-to-one"
            )
        measurements = _strict_string_tuple(
            self.source_measurement_hashes,
            "source_measurement_hashes",
            unique=False,
        )
        for name in (
            "pre_outcome_stream_weight_hash",
            "expected_ordered_route_axis_hash",
            "expected_measured_mask_hash",
            "expected_calibration_state_id",
        ):
            object.__setattr__(
                self,
                name,
                _strict_nonempty_string(getattr(self, name), name),
            )
        if type(self.schema) is not str or self.schema != _SOURCE_LEDGER_SCHEMA:
            raise ValueError("unsupported modern-v1 FD source-ledger schema")
        _require_bool(self.proposal_only, True, "proposal_only")
        _require_bool(
            self.claim_execution_certified,
            False,
            "claim_execution_certified",
        )
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "source_context_ids", sources)
        object.__setattr__(self, "source_row_hashes", rows)
        object.__setattr__(self, "source_measurement_hashes", measurements)

    @property
    def ledger_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fd-source-ledger-")

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1FDCudaEquivalenceReceipt:
    """Typed state-specific evidence for the CUDA algebraic implementation.

    This receipt records numerical implementation equivalence only.  It is not
    a scientific certificate, an execution authorization, or an owner
    signature.  Binding both the state and source-ledger hashes prevents a
    receipt for one stream from being replayed onto another.
    """

    source_state_hash: str
    source_ledger_hash: str
    ordered_route_axis_hash: str
    equivalence_evidence_hash: str
    canonical_algorithm_id: str = CANONICAL_FD_ALGORITHM_ID
    cuda_algorithm_id: str = CUDA_FD_ALGORITHM_ID
    algebraically_equivalent: bool = True
    proposal_only: bool = True
    claim_execution_certified: bool = False
    schema: str = _CUDA_EQUIVALENCE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_state_hash",
            "source_ledger_hash",
            "ordered_route_axis_hash",
            "equivalence_evidence_hash",
        ):
            object.__setattr__(
                self,
                name,
                _strict_nonempty_string(getattr(self, name), name),
            )
        if self.canonical_algorithm_id != CANONICAL_FD_ALGORITHM_ID:
            raise ValueError("CUDA receipt names the wrong canonical FD algorithm")
        if self.cuda_algorithm_id not in (
            CUDA_FD_ALGORITHM_ID,
            CUDA_GRAM_FD_ALGORITHM_ID,
        ):
            raise ValueError("CUDA receipt names the wrong producer FD algorithm")
        if type(self.schema) is not str or self.schema != _CUDA_EQUIVALENCE_SCHEMA:
            raise ValueError("unsupported CUDA FD equivalence-receipt schema")
        _require_bool(
            self.algebraically_equivalent,
            True,
            "algebraically_equivalent",
        )
        _require_bool(self.proposal_only, True, "proposal_only")
        _require_bool(
            self.claim_execution_certified,
            False,
            "claim_execution_certified",
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-fd-cuda-equivalence-",
        )

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True, init=False)
class V1FDAdapterResult:
    """Verified, proposal-only adapter bundle required by chart construction."""

    lineage: V1FrequentDirectionsLineage
    verification_receipt: V1FDLineageVerificationReceipt
    schema: str = _ADAPTER_RESULT_SCHEMA

    def __init__(
        self,
        *,
        lineage: V1FrequentDirectionsLineage,
        verification_receipt: V1FDLineageVerificationReceipt,
        _adapter_token: object,
    ) -> None:
        if _adapter_token is not _FD_ADAPTER_RESULT_TOKEN:
            raise TypeError("V1FDAdapterResult can only be issued by the modern-v1 FD adapter")
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "verification_receipt", verification_receipt)
        object.__setattr__(self, "schema", _ADAPTER_RESULT_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.lineage) is not V1FrequentDirectionsLineage:
            raise TypeError("FD adapter result requires an exact v1 lineage")
        if type(self.verification_receipt) is not V1FDLineageVerificationReceipt:
            raise TypeError("FD adapter result requires an exact verification receipt")
        if type(self.schema) is not str or self.schema != _ADAPTER_RESULT_SCHEMA:
            raise ValueError("unsupported modern-v1 FD adapter-result schema")
        if self.verification_receipt.lineage_hash != self.lineage.lineage_hash:
            raise ValueError("FD adapter receipt does not bind its lineage")

    @property
    def result_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fd-adapter-result-")

    @property
    def can_certify_decision(self) -> bool:
        return False


def _fresh_global_trust_state_hash(
    state: GlobalTrustGeometryState,
) -> str:
    """Recompute producer identity independently of its content-hash cache."""

    payload = {
        "schema": state.schema,
        "spec": state.spec,
        "route_axis_hash": ordered_string_axis_hash(
            state.route_ids,
            name="global trust route axis",
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


def _validate_state_snapshot(
    state: GlobalTrustGeometryState,
    ledger: V1FDSourceLedger,
) -> tuple[
    str,
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    GlobalTrustGeometryDerivationReceipt,
]:
    if type(state) is not GlobalTrustGeometryState:
        raise TypeError("adapter requires an exact GlobalTrustGeometryState")
    if type(state.spec) is not GlobalTrustGeometrySpec:
        raise TypeError("global trust state has a noncanonical spec")
    if state.schema != GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA:
        raise ValueError("unsupported global trust geometry state schema")

    arrays = (
        state.mean_component,
        state.centered_factor,
        state.residual_diagonal_bound,
    )
    if state.residual_diagonal_bound is None:
        raise ValueError("modern-v1 FD lineage requires the exact residual diagonal")
    if any(array is None or array.flags.writeable for array in arrays):
        raise ValueError("global trust state arrays must be immutable snapshots")

    route_hash = ordered_string_axis_hash(
        state.route_ids,
        name="global trust route axis",
    )
    if route_hash != ledger.expected_ordered_route_axis_hash:
        raise ValueError("state ordered route axis differs from the source ledger")
    expected_context_hash = content_hash(
        ledger.draw_ids,
        prefix="irg-trust-context-axis-",
    )
    if state.context_axis_hash != expected_context_hash:
        raise ValueError("state context-axis hash does not bind the ledger draw IDs")
    if state.context_count != len(ledger.draw_ids):
        raise ValueError("state context count differs from the source-ledger draws")
    if state.measured_mask_hash != ledger.expected_measured_mask_hash:
        raise ValueError("state measured mask differs from the source ledger")
    if state.calibration_state_id != ledger.expected_calibration_state_id:
        raise ValueError("state calibration identity differs from the source ledger")
    if tuple(state.source_measurement_hashes) != ledger.source_measurement_hashes:
        raise ValueError("state source_measurement_hashes differ from the exact source ledger")

    receipt = state.derivation_receipt
    if type(receipt) is not GlobalTrustGeometryDerivationReceipt:
        raise ValueError(
            "modern-v1 FD lineage requires an accumulator-produced "
            "derivation receipt; legacy or directly constructed states refuse"
        )
    if receipt.schema != GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA:
        raise ValueError("unsupported accumulator derivation-receipt schema")
    supported_producers = {
        CANONICAL_FD_ALGORITHM_ID: "canonical_numpy",
        CUDA_FD_ALGORITHM_ID: "torch_cuda",
        CUDA_GRAM_FD_ALGORITHM_ID: "gram_torch_cuda",
    }
    expected_implementation = supported_producers.get(receipt.producer_algorithm_id)
    if expected_implementation is None:
        raise ValueError("accumulator receipt names an unsupported FD algorithm")
    if receipt.producer_implementation != expected_implementation:
        raise ValueError("accumulator receipt implementation/algorithm mismatch")
    if receipt.trust_spec_hash != state.spec.content_hash:
        raise ValueError("accumulator receipt differs from the trust geometry spec")
    if receipt.ordered_route_axis_hash != route_hash:
        raise ValueError("accumulator receipt differs from the ordered route axis")
    if receipt.context_axis_hash != state.context_axis_hash:
        raise ValueError("accumulator receipt differs from the draw axis")
    if receipt.stream_entry_count != len(ledger.draw_ids):
        raise ValueError("accumulator source proof differs from the ledger draw count")
    if receipt.effective_context_count != state.context_count:
        raise ValueError("accumulator receipt differs from the effective context count")
    if receipt.stream_entry_count != receipt.effective_context_count:
        raise ValueError("modern-v1 FD source proof requires strictly positive-weight draws")
    expected_source_rows_hash = global_trust_ordered_source_rows_hash(ledger.source_row_hashes)
    if receipt.ordered_source_rows_hash != expected_source_rows_hash:
        raise ValueError("source-ledger row hashes do not match the accumulator-observed rows")
    if receipt.stream_weight_hash != ledger.pre_outcome_stream_weight_hash:
        raise ValueError("source-ledger weight hash does not match accumulator-observed weights")
    if receipt.measured_mask_hash != state.measured_mask_hash:
        raise ValueError("accumulator receipt differs from the measured mask")
    if receipt.calibration_state_id != state.calibration_state_id:
        raise ValueError("accumulator receipt differs from the calibration identity")
    if receipt.source_measurement_hashes != state.source_measurement_hashes:
        raise ValueError("accumulator receipt differs from source measurements")
    if receipt.total_weight != state.total_weight:
        raise ValueError("accumulator receipt differs from total stream weight")

    declared_state_hash = state.content_hash
    fresh_state_hash = _fresh_global_trust_state_hash(state)
    if declared_state_hash != fresh_state_hash:
        raise ValueError("global trust state changed after its content hash was observed")

    # Copy the small rank-by-route arrays only after identity validation.  The
    # copies close the derivation against concurrent source mutation and are
    # never retained by the returned lineage.
    mean = np.array(state.mean_component, dtype=np.float64, order="C", copy=True)
    centered = np.array(
        state.centered_factor,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    diagonal = np.array(
        state.residual_diagonal_bound,
        dtype=np.float64,
        order="C",
        copy=True,
    ).reshape(-1)
    if mean.shape[0] <= 0:
        raise ValueError("global trust effective factor requires mean rows")
    if diagonal.shape != (len(state.route_ids),):
        raise ValueError("exact residual diagonal has the wrong route axis")
    if receipt.mean_component_hash != content_hash(
        mean,
        prefix="irg-global-trust-derived-mean-",
    ):
        raise ValueError("accumulator receipt differs from the mean component")
    if receipt.centered_factor_hash != content_hash(
        centered,
        prefix="irg-global-trust-derived-centered-factor-",
    ):
        raise ValueError("accumulator receipt differs from the centered FD factor")
    if centered.shape[0] > receipt.sketch_rank:
        raise ValueError("centered FD factor exceeds the receipted accumulator rank")
    if receipt.exact_residual_diagonal_hash != content_hash(
        diagonal,
        prefix="irg-global-trust-derived-residual-diagonal-",
    ):
        raise ValueError("accumulator receipt differs from the exact residual diagonal")
    shrink = float(state.residual_spectral_bound)
    tolerance = 1e-10 * max(1.0, shrink)
    if np.any(diagonal > shrink + tolerance):
        raise ValueError("exact FD residual diagonal exceeds its spectral shrink envelope")
    expected_shrink = receipt.cumulative_fd_shrink / receipt.total_weight
    if not math.isclose(shrink, expected_shrink, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("state FD shrink violates its accumulator derivation algebra")
    decomposed_energy = math.fsum(
        (
            float(np.sum(mean * mean, dtype=np.float64)),
            float(np.sum(centered * centered, dtype=np.float64)),
            float(np.sum(diagonal, dtype=np.float64)),
        )
    )
    energy_tolerance = 1e-10 * max(
        1.0,
        receipt.normalized_stream_squared_frobenius,
    )
    if not math.isclose(
        decomposed_energy,
        receipt.normalized_stream_squared_frobenius,
        rel_tol=1e-10,
        abs_tol=energy_tolerance,
    ):
        raise ValueError("state factor/residual decomposition violates the streamed-energy proof")
    return declared_state_hash, route_hash, mean, centered, diagonal, receipt


def _implementation_from_metadata(
    state: GlobalTrustGeometryState,
    *,
    derivation_receipt: GlobalTrustGeometryDerivationReceipt,
    state_hash: str,
    route_axis_hash: str,
    source_ledger_hash: str,
    cuda_equivalence_receipt: V1FDCudaEquivalenceReceipt | None,
) -> tuple[str, str | None]:
    metadata = state.metadata
    algorithm = metadata.get("sketch_algorithm")
    if type(algorithm) is not str:
        raise ValueError("producer metadata lacks an exact FD algorithm identity")
    if algorithm != derivation_receipt.producer_algorithm_id:
        raise ValueError("producer metadata differs from the accumulator derivation receipt")
    if metadata.get("cross_module_terms_retained") is not True:
        raise ValueError("producer metadata does not retain cross-module terms")
    if metadata.get("spectral_residual_is_pathwise_upper_envelope") is not True:
        raise ValueError("producer metadata lacks the pathwise FD envelope")
    if metadata.get("residual_diagonal_is_exact_streamed_fd_remainder") is not True:
        raise ValueError("producer metadata does not attest an exact FD diagonal")

    if algorithm == CANONICAL_FD_ALGORITHM_ID:
        if (
            "canonical_numpy_algorithm_equivalent" in metadata
            or "host_context_route_rows_transferred" in metadata
        ):
            raise ValueError("canonical NumPy FD state carries forged CUDA producer metadata")
        if cuda_equivalence_receipt is not None:
            raise ValueError("canonical NumPy FD state must not carry a CUDA receipt")
        return "canonical_numpy", None
    if algorithm not in (CUDA_FD_ALGORITHM_ID, CUDA_GRAM_FD_ALGORITHM_ID):
        raise ValueError("unsupported or forged producer FD algorithm metadata")

    cuda_contracts = {
        CUDA_FD_ALGORITHM_ID: (
            "torch_cuda",
            CUDA_FD_IMPLEMENTATION_ID,
        ),
        CUDA_GRAM_FD_ALGORITHM_ID: (
            "gram_torch_cuda",
            CUDA_GRAM_FD_IMPLEMENTATION_ID,
        ),
    }
    expected_producer_implementation, lineage_implementation = cuda_contracts[algorithm]
    if derivation_receipt.producer_implementation != expected_producer_implementation:
        raise ValueError("CUDA derivation receipt has the wrong implementation identity")
    if metadata.get("canonical_numpy_algorithm_equivalent") is not True:
        raise ValueError("CUDA producer lacks its algebraic-equivalence marker")
    if metadata.get("host_context_route_rows_transferred") is not True:
        raise ValueError("CUDA source proof does not bind exact ordered source rows")
    proof_mode = metadata.get("derivation_proof_mode")
    if proof_mode == "ordered_host_snapshot_exact_v1":
        # Historical exact producer receipt.  It remains replayable, but it is
        # a whole-source host snapshot and therefore is not the bounded
        # performance path required by modern v1.
        if metadata.get("derivation_proof_full_tensor_host_snapshot") not in (None, True):
            raise ValueError("historical CUDA source proof snapshot metadata is inconsistent")
    elif proof_mode == "bounded_ordered_row_stream_exact_v1":
        if metadata.get("derivation_proof_full_tensor_host_snapshot") is not False:
            raise ValueError("bounded CUDA source proof cannot materialize a full host tensor")
        if metadata.get("derivation_proof_streaming_window_contexts") != 1:
            raise ValueError("bounded CUDA source proof must use a one-context proof window")
        peak_bytes = metadata.get("derivation_proof_peak_host_row_bytes")
        total_bytes = metadata.get("derivation_proof_total_host_transfer_bytes")
        if (
            type(peak_bytes) is not int
            or peak_bytes <= 0
            or type(total_bytes) is not int
            or total_bytes < peak_bytes
        ):
            raise ValueError("bounded CUDA source proof transfer accounting is invalid")
    else:
        raise ValueError("CUDA source proof mode is absent or unsupported")
    if cuda_equivalence_receipt is not None:
        raise ValueError(
            "caller-issued CUDA equivalence receipts cannot replace producer derivation proof"
        )
    # The producer receipt is recomputed above from the exact ordered rows,
    # weights, mean/factor/residual, shrink, and streamed energy.  It is the
    # only accepted CUDA equivalence identity; no caller boolean or tolerance
    # label participates in authorization.
    return lineage_implementation, derivation_receipt.receipt_hash


def _factor_payload_sha256(*arrays: np.ndarray) -> str:
    """Hash the row-wise concatenation without allocating a second factor."""

    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype=np.float64)
        if value.size:
            digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _effective_factor_hash(
    mean: np.ndarray,
    centered: np.ndarray,
) -> str:
    descriptor = {
        "dtype": "float64",
        "shape": [int(mean.shape[0] + centered.shape[0]), int(mean.shape[1])],
        "sha256": _factor_payload_sha256(mean, centered),
    }
    # This is exactly ``content_hash(np.vstack((mean, centered)))`` while
    # avoiding another potentially whole-route allocation.
    return content_hash(descriptor, prefix=_FACTOR_HASH_PREFIX)


def adapt_global_trust_geometry_to_v1_lineage(
    state: GlobalTrustGeometryState,
    *,
    source_ledger: V1FDSourceLedger,
    cuda_equivalence_receipt: V1FDCudaEquivalenceReceipt | None = None,
) -> V1FDAdapterResult:
    """Convert one exact producer state into inert modern-v1 FD lineage.

    No execution or certification switch exists on this surface.  The return
    value is a proposal-only verification bundle whose decision-certification
    properties are unconditionally false.
    """

    if type(source_ledger) is not V1FDSourceLedger:
        raise TypeError("adapter requires a typed V1FDSourceLedger")
    (
        state_hash,
        route_hash,
        mean,
        centered,
        diagonal,
        derivation_receipt,
    ) = _validate_state_snapshot(state, source_ledger)
    source_ledger_hash = source_ledger.ledger_hash
    implementation, equivalence_hash = _implementation_from_metadata(
        state,
        derivation_receipt=derivation_receipt,
        state_hash=state_hash,
        route_axis_hash=route_hash,
        source_ledger_hash=source_ledger_hash,
        cuda_equivalence_receipt=cuda_equivalence_receipt,
    )

    route_count = len(state.route_ids)
    mean_rows = int(mean.shape[0])
    centered_rows = int(centered.shape[0])
    effective_rows = mean_rows + centered_rows
    if effective_rows <= 0 or effective_rows > route_count:
        raise ValueError("effective mean-plus-centered FD factor rows exceed the route axis")

    factor_hash = _effective_factor_hash(mean, centered)
    mean_hash = content_hash(mean, prefix="partition-free-v1-fd-mean-")
    centered_hash = content_hash(
        centered,
        prefix="partition-free-v1-fd-centered-factor-",
    )
    diagonal_hash = content_hash(
        diagonal,
        prefix="partition-free-v1-fd-exact-residual-diagonal-",
    )
    factor_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_factor_derivation_receipt_v1",
            "derivation_id": EFFECTIVE_FACTOR_DERIVATION_ID,
            "source_state_hash": state_hash,
            "source_ledger_hash": source_ledger_hash,
            "accumulator_derivation_receipt_hash": (derivation_receipt.receipt_hash),
            "ordered_source_rows_hash": (derivation_receipt.ordered_source_rows_hash),
            "ordered_route_axis_hash": route_hash,
            "mean_component_hash": mean_hash,
            "centered_factor_hash": centered_hash,
            "effective_factor_hash": factor_hash,
            "mean_component_rows": mean_rows,
            "centered_fd_sketch_rank": centered_rows,
            "effective_factor_rows": effective_rows,
            "route_count": route_count,
        },
        prefix="partition-free-v1-fd-factor-receipt-",
    )

    fd_shrink = float(state.residual_spectral_bound)
    unnormalized_shrink = fd_shrink * float(state.total_weight)
    fd_shrink_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_shrink_derivation_receipt_v1",
            "derivation_id": FD_SHRINK_DERIVATION_ID,
            "source_state_hash": state_hash,
            "source_ledger_hash": source_ledger_hash,
            "normalized_fd_shrink": fd_shrink,
            "producer_total_weight": float(state.total_weight),
            "unnormalized_cumulative_fd_shrink": unnormalized_shrink,
            "ordered_route_axis_hash": route_hash,
        },
        prefix="partition-free-v1-fd-shrink-receipt-",
    )

    mean_energy = float(np.sum(mean * mean, dtype=np.float64))
    centered_energy = float(np.sum(centered * centered, dtype=np.float64))
    residual_trace = float(np.sum(diagonal, dtype=np.float64))
    exact_streamed_energy = math.fsum((mean_energy, centered_energy, residual_trace))
    if not math.isfinite(exact_streamed_energy) or exact_streamed_energy < 0.0:
        raise ValueError("derived exact streamed energy is invalid")
    exact_streamed_norm = math.sqrt(exact_streamed_energy)
    exact_norm_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_exact_norm_derivation_receipt_v1",
            "derivation_id": EXACT_STREAMED_NORM_DERIVATION_ID,
            "source_state_hash": state_hash,
            "source_ledger_hash": source_ledger_hash,
            "mean_component_hash": mean_hash,
            "centered_factor_hash": centered_hash,
            "exact_residual_diagonal_hash": diagonal_hash,
            "mean_squared_frobenius": mean_energy,
            "centered_factor_squared_frobenius": centered_energy,
            "exact_residual_trace": residual_trace,
            "exact_streamed_squared_frobenius": exact_streamed_energy,
            "exact_streamed_frobenius_norm": exact_streamed_norm,
            "ordered_route_axis_hash": route_hash,
        },
        prefix="partition-free-v1-fd-exact-norm-receipt-",
    )

    lineage = V1FrequentDirectionsLineage(
        algorithm_id=CANONICAL_FD_ALGORITHM_ID,
        implementation=implementation,
        route_axis_hash=route_hash,
        measured_mask_hash=state.measured_mask_hash,
        draw_ids=source_ledger.draw_ids,
        source_context_ids=source_ledger.source_context_ids,
        source_row_hashes=source_ledger.source_row_hashes,
        stream_weight_hash=source_ledger.pre_outcome_stream_weight_hash,
        accumulator_context_axis_hash=state.context_axis_hash,
        factor_hash=factor_hash,
        factor_receipt_hash=factor_receipt_hash,
        fd_shrink_receipt_hash=fd_shrink_receipt_hash,
        exact_streamed_norm_receipt_hash=exact_norm_receipt_hash,
        # ``V1FrequentDirectionsLineage.fd_rank`` predates this adapter and is
        # a row-capacity field, not an algebraic-rank assertion.  The effective
        # H factor includes both the mean and centered sketch rows.
        fd_rank=effective_rows,
        factor_row_count=effective_rows,
        route_count=route_count,
        fd_shrink=fd_shrink,
        exact_streamed_norm=exact_streamed_norm,
        canonical_equivalence_receipt_hash=equivalence_hash,
        metadata={
            "adapter_schema": _ADAPTER_SCHEMA,
            "proposal_only": True,
            "claim_execution_certified": False,
            "source_state_hash": state_hash,
            "source_ledger_hash": source_ledger_hash,
            "ordered_route_axis_hash": route_hash,
            "producer_algorithm_id": state.metadata["sketch_algorithm"],
            "factor_derivation_id": EFFECTIVE_FACTOR_DERIVATION_ID,
            "fd_shrink_derivation_id": FD_SHRINK_DERIVATION_ID,
            "exact_streamed_norm_derivation_id": (EXACT_STREAMED_NORM_DERIVATION_ID),
            "exact_residual_diagonal_hash": diagonal_hash,
            "residual_diagonal_status": "proposal_only",
            "centered_fd_sketch_rank": centered_rows,
            "mean_component_rows": mean_rows,
            "effective_factor_rows": effective_rows,
            "fd_rank_semantics": "effective_factor_row_capacity_v1",
            "producer_spec_initial_rank": int(state.spec.initial_rank),
            "producer_spec_maximum_rank": int(state.spec.maximum_rank),
        },
    )
    if lineage.can_certify_decision:  # pragma: no cover - permanent firewall
        raise RuntimeError("modern-v1 FD adapter crossed its proposal firewall")
    verified_algebra_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_verified_algebra_v1",
            "source_state_hash": state_hash,
            "source_ledger_hash": source_ledger_hash,
            "accumulator_derivation_receipt_hash": (derivation_receipt.receipt_hash),
            "lineage_hash": lineage.lineage_hash,
            "ordered_route_axis_hash": route_hash,
            "ordered_source_rows_hash": (derivation_receipt.ordered_source_rows_hash),
            "stream_weight_hash": derivation_receipt.stream_weight_hash,
            "mean_component_hash": mean_hash,
            "centered_factor_hash": centered_hash,
            "exact_residual_diagonal_hash": diagonal_hash,
            "factor_hash": factor_hash,
            "fd_shrink": fd_shrink,
            "exact_streamed_norm": exact_streamed_norm,
            "factor_rows": effective_rows,
            "route_count": route_count,
        },
        prefix="partition-free-v1-fd-verified-algebra-",
    )
    verification_receipt = V1FDLineageVerificationReceipt(
        lineage_hash=lineage.lineage_hash,
        source_state_hash=state_hash,
        source_ledger_hash=source_ledger_hash,
        accumulator_derivation_receipt_hash=derivation_receipt.receipt_hash,
        ordered_source_rows_hash=derivation_receipt.ordered_source_rows_hash,
        stream_weight_hash=derivation_receipt.stream_weight_hash,
        factor_derivation_receipt_hash=factor_receipt_hash,
        fd_shrink_derivation_receipt_hash=fd_shrink_receipt_hash,
        exact_norm_derivation_receipt_hash=exact_norm_receipt_hash,
        verified_algebra_hash=verified_algebra_hash,
    )
    return V1FDAdapterResult(
        lineage=lineage,
        verification_receipt=verification_receipt,
        _adapter_token=_FD_ADAPTER_RESULT_TOKEN,
    )


__all__ = [
    "CANONICAL_FD_ALGORITHM_ID",
    "CUDA_FD_IMPLEMENTATION_ID",
    "CUDA_GRAM_FD_ALGORITHM_ID",
    "CUDA_GRAM_FD_IMPLEMENTATION_ID",
    "CUDA_FD_ALGORITHM_ID",
    "EFFECTIVE_FACTOR_DERIVATION_ID",
    "EXACT_STREAMED_NORM_DERIVATION_ID",
    "FD_SHRINK_DERIVATION_ID",
    "V1FDCudaEquivalenceReceipt",
    "V1FDAdapterResult",
    "V1FDSourceLedger",
    "adapt_global_trust_geometry_to_v1_lineage",
]
