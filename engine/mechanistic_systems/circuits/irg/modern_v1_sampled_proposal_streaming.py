"""Raw-logit provider bridges for sampled modern-v1 discovery.

The optimized scorer's loss-only fast path is intentionally nominal: a sink
must inherit ``_RouteChunkedLossOnlyConsumer`` before it may receive the raw
positive-removal stream.  This module contains the two sampled-discovery
adapters that are allowed across that boundary:

* a zero-row-retention adapter around the GPU-reduced corrected-attribution
  packet consumer; and
* a bounded one-packet ``[draw, route]`` staging adapter for the separate-
  stratum Frequent-Directions response geometry.

Legacy v1 requests accept the scorer-produced ``-G*phi*c`` stream in both
children and remain replay-only.  Version-2 requests type the reduced child as
corrected Aumann--Shapley evidence and the FD child as the literal positive-
removal relaxed-gate Jacobian ``-G*1[a>0]*c``.  The adapters deliberately carry
discovery-input authority only.  In particular, none of their receipts can
certify an F/Q budget, open a BCE oracle, or commit a mask; canonical-Q input
authority arises only from the enclosing same-capture/same-VJP transaction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import time
from types import MappingProxyType
from typing import Any, Callable, Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_proposal_packet_gpu import (
    V1GPUReducedProposalPacket,
    V1GPUReducedProposalPacketConsumer,
    V1GPUReducedProposalPacketRequest,
)
from mechanistic_systems.circuits.irg.modern_v1_stratified_fd_gpu import (
    V1StratifiedDeviceFDTrustAccumulator,
    V1StratifiedFDDesign,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)
from mechanistic_systems.measurement.optimized_irg_providers import (
    _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
    _RouteChunkedLossOnlyConsumer,
)

try:  # pragma: no cover - import availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA: Final = (
    "partition_free_v1_sampled_reduced_packet_stream_request_v1"
)
SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_corrected_evidence_stream_request_v2"
)
SAMPLED_FD_STREAM_REQUEST_SCHEMA: Final = (
    "partition_free_v1_sampled_fd_packet_stream_request_v1"
)
SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_literal_jacobian_fd_stream_request_v2"
)
SAMPLED_FD_PACKET_MEMORY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_sampled_fd_packet_memory_forecast_v1"
)
SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_reduced_packet_stream_receipt_v1"
)
SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_corrected_evidence_stream_receipt_v2"
)
SAMPLED_FD_STREAM_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_fd_packet_stream_receipt_v1"
)
SAMPLED_FD_STREAM_RECEIPT_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_literal_jacobian_fd_stream_receipt_v2"
)
SAMPLED_FD_APPEND_RESULT_SCHEMA: Final = (
    "partition_free_v1_sampled_fd_packet_append_result_v1"
)
SAMPLED_DISCOVERY_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_sampled_proposal_discovery_closure_v1"
)

RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE: Final = (
    _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
)
CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS: Final = (
    "corrected_aumann_shapley_positive_removal_negative_G_phi_c"
)
LITERAL_GATE_JACOBIAN_ROW_SEMANTICS: Final = (
    "literal_positive_removal_gate_jacobian_negative_G_indicator_a_positive_c"
)
SAMPLED_DISCOVERY_AUTHORITY_SCOPE: Final = (
    "discovery_input_only_no_fq_oracle_or_mask_commit_v1"
)
SAMPLED_FD_PACKET_MEMORY_SCOPE: Final = (
    "adapter_full_packet_staging_plus_colive_scorer_chunk_excluding_"
    "accumulator_state_allocator_overhead_and_library_workspace_v1"
)

_REDUCED_REQUEST_PREFIX: Final = "partition-free-v1-sampled-reduced-request-"
_FD_REQUEST_PREFIX: Final = "partition-free-v1-sampled-fd-request-"
_FD_MEMORY_PREFIX: Final = "partition-free-v1-sampled-fd-memory-"
_REDUCED_RECEIPT_PREFIX: Final = "partition-free-v1-sampled-reduced-receipt-"
_FD_RECEIPT_PREFIX: Final = "partition-free-v1-sampled-fd-receipt-"
_FD_RESULT_PREFIX: Final = "partition-free-v1-sampled-fd-append-result-"
_CLOSURE_PREFIX: Final = "partition-free-v1-sampled-discovery-closure-"
_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-sampled-context-axis-"
_SOURCE_AXIS_PREFIX: Final = "partition-free-v1-sampled-source-axis-"
_CHILD_LEDGER_PREFIX: Final = "partition-free-v1-sampled-fd-child-ledger-"
_FLOAT64_BYTES: Final = 8


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _identity_axis(
    values: Sequence[str],
    *,
    name: str,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(_identity(value, name=f"{name} item") for value in values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _canonical_device(value: Any) -> Any:
    if torch is None:
        raise RuntimeError("sampled proposal streaming requires PyTorch")
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        try:
            return torch.device("cuda", int(torch.cuda.current_device()))
        except (AssertionError, RuntimeError):
            return device
    return device


def _declared_device(value: object) -> str:
    text = _identity(value, name="declared device")
    if torch is None:
        raise RuntimeError("sampled proposal streaming requires PyTorch")
    device = torch.device(text)
    if device.type == "cuda" and device.index is None:
        raise ValueError("sampled stream CUDA devices must declare an explicit index")
    return str(device)


def _device_identity(device: Any) -> str:
    runtime = _canonical_device(device)
    if runtime.type != "cuda":
        return str(runtime)
    try:
        name = torch.cuda.get_device_name(runtime)
    except (AssertionError, RuntimeError):
        name = "cuda-device-name-unavailable"
    return f"cuda:{runtime.index}:{name}"


def _validated_contiguous_route_windows(
    route_windows: Sequence[Sequence[int]],
    *,
    route_count: int,
    chunk_size: int,
    name: str,
) -> tuple[tuple[int, int], ...]:
    """Validate an exact ordered route cover with bounded chunk width.

    ``registered_route_chunk_size`` is the sealed maximum working width, not
    a promise that model route blocks divide into that fixed-width tiling.
    Production scorers flush at route-block boundaries.  The packetization
    contract therefore requires the physical windows to cover the canonical
    route axis exactly once, in ascending order, with every width bounded by
    the registered maximum.  The realized windows are retained in the typed,
    content-addressed receipt.
    """

    routes = _positive_int(route_count, name="route count")
    chunk = _positive_int(chunk_size, name="registered route chunk size")
    if isinstance(route_windows, (str, bytes)):
        raise TypeError(f"{name} route windows must be a sequence of pairs")
    windows: list[tuple[int, int]] = []
    for index, raw_window in enumerate(route_windows):
        if isinstance(raw_window, (str, bytes)):
            raise TypeError(f"{name} route window {index} must be one pair")
        window = tuple(raw_window)
        if len(window) != 2:
            raise ValueError(f"{name} route window {index} must contain two values")
        windows.append(
            (
                _nonnegative_int(
                    window[0],
                    name=f"{name} route window {index} start",
                ),
                _positive_int(
                    window[1],
                    name=f"{name} route window {index} stop",
                ),
            )
        )
    cursor = 0
    for start, stop in windows:
        width = stop - start
        if start != cursor or width <= 0 or width > chunk or stop > routes:
            raise ValueError(f"{name} route windows are incomplete")
        cursor = stop
    if cursor != routes:
        raise ValueError(f"{name} route windows are incomplete")
    return tuple(windows)


def _assert_route_axis_matches(
    observed: Sequence[str],
    expected: Sequence[str],
    *,
    name: str,
) -> None:
    if observed is expected:
        return
    if len(observed) != len(expected) or ordered_string_axis_hash(
        observed,
        name=f"{name} observed route axis",
    ) != ordered_string_axis_hash(
        expected,
        name=f"{name} expected route axis",
    ):
        raise ValueError(f"{name} route axis differs from its frozen request")


@dataclass(frozen=True, slots=True)
class V1SampledReducedPacketStreamRequest:
    """Frozen provider identity for one GPU-reduced discovery packet."""

    packet_request: V1GPUReducedProposalPacketRequest
    route_ids: tuple[str, ...]
    destination_transport_hash: str
    raw_logit_stream_identity_hash: str
    declared_device: str
    raw_target_logit_row_semantics: str | None = None
    positive_removal_stream_coordinate: str = (
        RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
    )
    schema: str = SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.packet_request) is not V1GPUReducedProposalPacketRequest:
            raise TypeError("sampled reduced stream requires its exact packet request")
        self.packet_request.assert_integrity()
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="sampled reduced stream route axis",
        )
        destination = _identity(
            self.destination_transport_hash,
            name="destination transport hash",
        )
        stream = _identity(
            self.raw_logit_stream_identity_hash,
            name="raw-logit stream identity hash",
        )
        device = _declared_device(self.declared_device)
        if len(routes) != self.packet_request.route_count or ordered_string_axis_hash(
            routes,
            name="sampled reduced stream route axis",
        ) != self.packet_request.route_axis_hash:
            raise ValueError("sampled reduced stream route axis differs from packet")
        if stream != self.packet_request.raw_logit_stream_source_hash:
            raise ValueError("sampled reduced stream identity differs from packet")
        if (
            self.positive_removal_stream_coordinate
            != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        ):
            raise ValueError("sampled reduced stream accepts only raw target logits")
        if self.schema == SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA:
            if self.raw_target_logit_row_semantics is not None:
                raise ValueError(
                    "legacy sampled reduced requests cannot claim canonical "
                    "corrected-evidence semantics"
                )
        elif self.schema == SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2:
            if (
                self.raw_target_logit_row_semantics
                != CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
            ):
                raise ValueError(
                    "canonical sampled reduced requests require corrected "
                    "Aumann--Shapley evidence semantics"
                )
        else:
            raise ValueError("sampled reduced stream request schema is invalid")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "destination_transport_hash", destination)
        object.__setattr__(self, "raw_logit_stream_identity_hash", stream)
        object.__setattr__(self, "declared_device", device)

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "packet_request_hash": self.packet_request.request_hash,
            "packetization_identity_hash": (
                self.packet_request.packetization_identity_hash
            ),
            "route_axis_hash": self.packet_request.route_axis_hash,
            "route_count": len(self.route_ids),
            "destination_transport_hash": self.destination_transport_hash,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "declared_device": self.declared_device,
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
        }
        if self.schema == SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2:
            payload["raw_target_logit_row_semantics"] = (
                self.raw_target_logit_row_semantics
            )
        return payload

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.packet_request.assert_integrity()
        observed = content_hash(self._hash_payload(), prefix=_REDUCED_REQUEST_PREFIX)
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("sampled reduced stream request was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("sampled reduced stream request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledFDPacketMemoryForecast:
    """Exact bulk-byte admission ledger for the one-packet FD bridge."""

    route_count: int
    packet_draw_count: int
    registered_route_chunk_size: int
    adapter_staging_device_bytes: int
    colive_scorer_chunk_device_bytes: int
    joint_peak_device_bytes: int
    packet_draw_memory_budget_bytes: int
    host_context_route_bytes: int = 0
    accumulator_state_device_bytes_in_scope: bool = False
    admitted: bool = True
    memory_accounting_scope: str = SAMPLED_FD_PACKET_MEMORY_SCOPE
    schema: str = SAMPLED_FD_PACKET_MEMORY_FORECAST_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="route count")
        draws = _positive_int(self.packet_draw_count, name="packet draw count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        if chunk > routes:
            raise ValueError("registered route chunk size exceeds route axis")
        budget = _nonnegative_int(
            self.packet_draw_memory_budget_bytes,
            name="packet-draw memory budget bytes",
        )
        expected_staging = draws * routes * _FLOAT64_BYTES
        expected_chunk = draws * chunk * _FLOAT64_BYTES
        expected_peak = expected_staging + expected_chunk
        checks = {
            "adapter_staging_device_bytes": expected_staging,
            "colive_scorer_chunk_device_bytes": expected_chunk,
            "joint_peak_device_bytes": expected_peak,
            "host_context_route_bytes": 0,
        }
        for name, expected in checks.items():
            if _nonnegative_int(getattr(self, name), name=name) != expected:
                raise ValueError(f"{name} does not replay from forecast inputs")
        if self.accumulator_state_device_bytes_in_scope is not False:
            raise ValueError("packet forecast cannot absorb FD accumulator state")
        expected_admitted = expected_peak <= budget
        if self.admitted is not expected_admitted:
            raise ValueError("packet forecast admission flag does not replay")
        if self.memory_accounting_scope != SAMPLED_FD_PACKET_MEMORY_SCOPE:
            raise ValueError("sampled FD packet memory scope is invalid")
        if self.schema != SAMPLED_FD_PACKET_MEMORY_FORECAST_SCHEMA:
            raise ValueError("sampled FD packet memory schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "packet_draw_count", draws)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "packet_draw_memory_budget_bytes", budget)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "route_count": self.route_count,
            "packet_draw_count": self.packet_draw_count,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "adapter_staging_device_bytes": self.adapter_staging_device_bytes,
            "colive_scorer_chunk_device_bytes": (
                self.colive_scorer_chunk_device_bytes
            ),
            "joint_peak_device_bytes": self.joint_peak_device_bytes,
            "packet_draw_memory_budget_bytes": (
                self.packet_draw_memory_budget_bytes
            ),
            "host_context_route_bytes": self.host_context_route_bytes,
            "accumulator_state_device_bytes_in_scope": (
                self.accumulator_state_device_bytes_in_scope
            ),
            "admitted": self.admitted,
            "memory_accounting_scope": self.memory_accounting_scope,
        }

    @property
    def forecast_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_FD_MEMORY_PREFIX)
        if self._forecast_hash is not None and observed != self._forecast_hash:
            raise ValueError("sampled FD packet memory forecast was tampered")
        object.__setattr__(self, "_forecast_hash", observed)
        return observed

    def assert_admissible(self) -> None:
        if not self.admitted:
            raise MemoryError(
                "sampled FD packet draw exceeds its explicit device-memory budget"
            )


def forecast_sampled_fd_packet_memory(
    *,
    route_count: int,
    packet_draw_count: int,
    registered_route_chunk_size: int,
    packet_draw_memory_budget_bytes: int,
) -> V1SampledFDPacketMemoryForecast:
    routes = _positive_int(route_count, name="route count")
    draws = _positive_int(packet_draw_count, name="packet draw count")
    chunk = _positive_int(
        registered_route_chunk_size,
        name="registered route chunk size",
    )
    if chunk > routes:
        raise ValueError("registered route chunk size exceeds route axis")
    budget = _nonnegative_int(
        packet_draw_memory_budget_bytes,
        name="packet-draw memory budget bytes",
    )
    staging = draws * routes * _FLOAT64_BYTES
    scorer_chunk = draws * chunk * _FLOAT64_BYTES
    peak = staging + scorer_chunk
    return V1SampledFDPacketMemoryForecast(
        route_count=routes,
        packet_draw_count=draws,
        registered_route_chunk_size=chunk,
        adapter_staging_device_bytes=staging,
        colive_scorer_chunk_device_bytes=scorer_chunk,
        joint_peak_device_bytes=peak,
        packet_draw_memory_budget_bytes=budget,
        admitted=peak <= budget,
    )


@dataclass(frozen=True, slots=True)
class V1SampledFDStreamRequest:
    """Frozen identity and admission budget for one same-stratum FD packet."""

    design: V1StratifiedFDDesign
    context_ids: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    source_measurement_hashes: tuple[str, ...]
    destination_transport_hash: str
    raw_logit_stream_identity_hash: str
    packet_index: int
    registered_route_chunk_size: int
    packet_draw_memory_budget_bytes: int
    declared_device: str
    raw_target_logit_row_semantics: str | None = None
    positive_removal_stream_coordinate: str = (
        RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
    )
    schema: str = SAMPLED_FD_STREAM_REQUEST_SCHEMA
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.design) is not V1StratifiedFDDesign:
            raise TypeError("sampled FD stream requires its exact design type")
        self.design.assert_integrity()
        contexts = _identity_axis(
            self.context_ids,
            name="sampled FD context ledger",
            unique=True,
        )
        labels = _identity_axis(
            self.stratum_labels,
            name="sampled FD stratum ledger",
            unique=False,
        )
        sources = _identity_axis(
            self.source_measurement_hashes,
            name="sampled FD source-measurement ledger",
            unique=False,
        )
        if not (len(contexts) == len(labels) == len(sources)):
            raise ValueError("sampled FD packet ledgers must align")
        if len(set(labels)) != 1:
            raise ValueError("sampled FD packet cannot mix strata")
        if labels[0] not in self.design.stratum_names:
            raise ValueError("sampled FD packet names an unregistered stratum")
        destination = _identity(
            self.destination_transport_hash,
            name="destination transport hash",
        )
        stream = _identity(
            self.raw_logit_stream_identity_hash,
            name="raw-logit stream identity hash",
        )
        if stream != self.design.raw_logit_stream_source_hash:
            raise ValueError("sampled FD stream identity differs from design")
        packet_index = _nonnegative_int(self.packet_index, name="packet index")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        if chunk > len(self.design.route_ids):
            raise ValueError("registered route chunk size exceeds route axis")
        budget = _nonnegative_int(
            self.packet_draw_memory_budget_bytes,
            name="packet-draw memory budget bytes",
        )
        device = _declared_device(self.declared_device)
        if (
            self.positive_removal_stream_coordinate
            != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        ):
            raise ValueError("sampled FD stream accepts only raw target logits")
        if self.schema == SAMPLED_FD_STREAM_REQUEST_SCHEMA:
            if self.raw_target_logit_row_semantics is not None:
                raise ValueError(
                    "legacy sampled FD requests cannot claim literal-Jacobian "
                    "semantics"
                )
        elif self.schema == SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2:
            if (
                self.raw_target_logit_row_semantics
                != LITERAL_GATE_JACOBIAN_ROW_SEMANTICS
            ):
                raise ValueError(
                    "canonical sampled FD requests require literal positive-"
                    "removal gate-Jacobian semantics"
                )
        else:
            raise ValueError("sampled FD stream request schema is invalid")
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "stratum_labels", labels)
        object.__setattr__(self, "source_measurement_hashes", sources)
        object.__setattr__(self, "destination_transport_hash", destination)
        object.__setattr__(self, "raw_logit_stream_identity_hash", stream)
        object.__setattr__(self, "packet_index", packet_index)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "packet_draw_memory_budget_bytes", budget)
        object.__setattr__(self, "declared_device", device)

    @property
    def memory_forecast(self) -> V1SampledFDPacketMemoryForecast:
        return forecast_sampled_fd_packet_memory(
            route_count=len(self.design.route_ids),
            packet_draw_count=len(self.context_ids),
            registered_route_chunk_size=self.registered_route_chunk_size,
            packet_draw_memory_budget_bytes=self.packet_draw_memory_budget_bytes,
        )

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "design_hash": self.design.design_hash,
            "context_ids": self.context_ids,
            "stratum_labels": self.stratum_labels,
            "source_measurement_hashes": self.source_measurement_hashes,
            "destination_transport_hash": self.destination_transport_hash,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "packet_index": self.packet_index,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "packet_draw_memory_forecast_hash": self.memory_forecast.forecast_hash,
            "declared_device": self.declared_device,
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
        }
        if self.schema == SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2:
            payload["raw_target_logit_row_semantics"] = (
                self.raw_target_logit_row_semantics
            )
        return payload

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.design.assert_integrity()
        observed = content_hash(self._hash_payload(), prefix=_FD_REQUEST_PREFIX)
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("sampled FD stream request was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("sampled FD stream request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledFDAppendResult:
    """Small typed child result for one accepted FD packet boundary."""

    request_hash: str
    design_hash: str
    stratum_name: str
    packet_boundary_index_before: int
    packet_boundary_index_after: int
    target_stratum_context_count_after: int
    context_ledger_hash_after: str
    source_measurement_ledger_hash_after: str
    latest_source_row_hashes: tuple[str, ...]
    full_accumulator_snapshot_emitted: bool = False
    source_row_proof_authority: bool = False
    schema: str = SAMPLED_FD_APPEND_RESULT_SCHEMA
    _result_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "design_hash",
            "stratum_name",
            "context_ledger_hash_after",
            "source_measurement_ledger_hash_after",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        before = _nonnegative_int(
            self.packet_boundary_index_before,
            name="packet boundary before",
        )
        after = _positive_int(
            self.packet_boundary_index_after,
            name="packet boundary after",
        )
        if after != before + 1:
            raise ValueError(
                "sampled FD append must advance exactly one packet boundary"
            )
        count = _positive_int(
            self.target_stratum_context_count_after,
            name="target-stratum context count after",
        )
        hashes = tuple(
            _identity(value, name="latest source-row hash")
            for value in self.latest_source_row_hashes
        )
        if self.full_accumulator_snapshot_emitted is not False:
            raise ValueError("sampled FD packet append cannot emit a full snapshot")
        if self.source_row_proof_authority is not False:
            raise ValueError(
                "sampled FD append result cannot claim row-proof authority"
            )
        if self.schema != SAMPLED_FD_APPEND_RESULT_SCHEMA:
            raise ValueError("sampled FD append result schema is invalid")
        object.__setattr__(self, "packet_boundary_index_before", before)
        object.__setattr__(self, "packet_boundary_index_after", after)
        object.__setattr__(self, "target_stratum_context_count_after", count)
        object.__setattr__(self, "latest_source_row_hashes", hashes)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "request_hash": self.request_hash,
            "design_hash": self.design_hash,
            "stratum_name": self.stratum_name,
            "packet_boundary_index_before": self.packet_boundary_index_before,
            "packet_boundary_index_after": self.packet_boundary_index_after,
            "target_stratum_context_count_after": (
                self.target_stratum_context_count_after
            ),
            "context_ledger_hash_after": self.context_ledger_hash_after,
            "source_measurement_ledger_hash_after": (
                self.source_measurement_ledger_hash_after
            ),
            "latest_source_row_hashes": self.latest_source_row_hashes,
            "full_accumulator_snapshot_emitted": (
                self.full_accumulator_snapshot_emitted
            ),
            "source_row_proof_authority": self.source_row_proof_authority,
        }

    @property
    def result_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_FD_RESULT_PREFIX)
        if self._result_hash is not None and observed != self._result_hash:
            raise ValueError("sampled FD append result was tampered")
        object.__setattr__(self, "_result_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledProposalDiscoveryClosure:
    """Authority-limited closure shared by both sampled stream adapters."""

    adapter_kind: str
    adapter_request_hash: str
    adapter_receipt_hash: str
    child_contract_hash: str
    child_result_hash: str
    child_result_schema: str
    raw_logit_stream_identity_hash: str
    positive_removal_stream_coordinate: str = (
        RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
    )
    authority_scope: str = SAMPLED_DISCOVERY_AUTHORITY_SCOPE
    proposal_discovery_input_authority: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_DISCOVERY_CLOSURE_SCHEMA
    _closure_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.adapter_kind not in {
            "gpu_reduced_corrected_attribution_packet",
            "same_stratum_fd_packet_append",
        }:
            raise ValueError("sampled discovery adapter kind is invalid")
        for name in (
            "adapter_request_hash",
            "adapter_receipt_hash",
            "child_contract_hash",
            "child_result_hash",
            "child_result_schema",
            "raw_logit_stream_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.positive_removal_stream_coordinate
            != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            or self.authority_scope != SAMPLED_DISCOVERY_AUTHORITY_SCOPE
            or self.proposal_discovery_input_authority is not True
            or any(
                (
                    self.fq_certification_authority,
                    self.bce_oracle_authority,
                    self.mask_commit_authority,
                )
            )
        ):
            raise ValueError("sampled discovery closure exceeded its authority")
        if self.schema != SAMPLED_DISCOVERY_CLOSURE_SCHEMA:
            raise ValueError("sampled discovery closure schema is invalid")

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "adapter_kind": self.adapter_kind,
            "adapter_request_hash": self.adapter_request_hash,
            "adapter_receipt_hash": self.adapter_receipt_hash,
            "child_contract_hash": self.child_contract_hash,
            "child_result_hash": self.child_result_hash,
            "child_result_schema": self.child_result_schema,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
            "authority_scope": self.authority_scope,
            "proposal_discovery_input_authority": (
                self.proposal_discovery_input_authority
            ),
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def closure_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_CLOSURE_PREFIX)
        if self._closure_hash is not None and observed != self._closure_hash:
            raise ValueError("sampled discovery closure was tampered")
        object.__setattr__(self, "_closure_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledReducedStreamReceipt:
    request_hash: str
    child_request_hash: str
    child_packet_hash: str
    child_derivation_receipt_hash: str
    route_axis_hash: str
    context_axis_hash: str
    destination_transport_hash: str
    raw_logit_stream_identity_hash: str
    device_identity: str
    route_count: int
    registered_route_chunk_size: int
    route_windows: tuple[tuple[int, int], ...]
    shared_tensor_observation_count: int
    raw_target_logit_row_semantics: str | None = None
    authority_scope: str = SAMPLED_DISCOVERY_AUTHORITY_SCOPE
    schema: str = SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "child_request_hash",
            "child_packet_hash",
            "child_derivation_receipt_hash",
            "route_axis_hash",
            "context_axis_hash",
            "destination_transport_hash",
            "raw_logit_stream_identity_hash",
            "device_identity",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        windows = _validated_contiguous_route_windows(
            self.route_windows,
            route_count=routes,
            chunk_size=chunk,
            name="sampled reduced receipt",
        )
        observations = _positive_int(
            self.shared_tensor_observation_count,
            name="shared tensor observation count",
        )
        if observations != len(windows):
            raise ValueError("sampled reduced receipt chunk count does not replay")
        if self.schema == SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA:
            if self.raw_target_logit_row_semantics is not None:
                raise ValueError(
                    "legacy sampled reduced receipts cannot claim canonical "
                    "corrected-evidence semantics"
                )
        elif self.schema == SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA_V2:
            if (
                self.raw_target_logit_row_semantics
                != CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
            ):
                raise ValueError(
                    "canonical sampled reduced receipts require corrected "
                    "Aumann--Shapley evidence semantics"
                )
        else:
            raise ValueError("sampled reduced receipt schema is invalid")
        if self.authority_scope != SAMPLED_DISCOVERY_AUTHORITY_SCOPE:
            raise ValueError("sampled reduced receipt authority is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "route_windows", windows)
        object.__setattr__(self, "shared_tensor_observation_count", observations)

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "request_hash",
                "child_request_hash",
                "child_packet_hash",
                "child_derivation_receipt_hash",
                "route_axis_hash",
                "context_axis_hash",
                "destination_transport_hash",
                "raw_logit_stream_identity_hash",
                "device_identity",
                "route_count",
                "registered_route_chunk_size",
                "route_windows",
                "shared_tensor_observation_count",
                "authority_scope",
            )
        }
        if self.schema == SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA_V2:
            payload["raw_target_logit_row_semantics"] = (
                self.raw_target_logit_row_semantics
            )
        return payload

    @property
    def receipt_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_REDUCED_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("sampled reduced stream receipt was tampered")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledFDStreamReceipt:
    request_hash: str
    design_hash: str
    append_result_hash: str
    route_axis_hash: str
    context_axis_hash: str
    stratum_axis_hash: str
    source_measurement_axis_hash: str
    destination_transport_hash: str
    raw_logit_stream_identity_hash: str
    device_identity: str
    memory_forecast_hash: str
    route_count: int
    registered_route_chunk_size: int
    route_windows: tuple[tuple[int, int], ...]
    packet_boundary_index_before: int
    packet_boundary_index_after: int
    shared_tensor_observation_count: int
    raw_target_logit_row_semantics: str | None = None
    full_accumulator_snapshot_emitted: bool = False
    authority_scope: str = SAMPLED_DISCOVERY_AUTHORITY_SCOPE
    schema: str = SAMPLED_FD_STREAM_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "design_hash",
            "append_result_hash",
            "route_axis_hash",
            "context_axis_hash",
            "stratum_axis_hash",
            "source_measurement_axis_hash",
            "destination_transport_hash",
            "raw_logit_stream_identity_hash",
            "device_identity",
            "memory_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        windows = _validated_contiguous_route_windows(
            self.route_windows,
            route_count=routes,
            chunk_size=chunk,
            name="sampled FD receipt",
        )
        before = _nonnegative_int(
            self.packet_boundary_index_before,
            name="packet boundary before",
        )
        after = _positive_int(
            self.packet_boundary_index_after,
            name="packet boundary after",
        )
        if after != before + 1:
            raise ValueError("sampled FD receipt did not append exactly one packet")
        observations = _positive_int(
            self.shared_tensor_observation_count,
            name="shared tensor observation count",
        )
        if observations != len(windows):
            raise ValueError("sampled FD receipt chunk count does not replay")
        if self.schema == SAMPLED_FD_STREAM_RECEIPT_SCHEMA:
            if self.raw_target_logit_row_semantics is not None:
                raise ValueError(
                    "legacy sampled FD receipts cannot claim literal-Jacobian "
                    "semantics"
                )
        elif self.schema == SAMPLED_FD_STREAM_RECEIPT_SCHEMA_V2:
            if (
                self.raw_target_logit_row_semantics
                != LITERAL_GATE_JACOBIAN_ROW_SEMANTICS
            ):
                raise ValueError(
                    "canonical sampled FD receipts require literal positive-"
                    "removal gate-Jacobian semantics"
                )
        else:
            raise ValueError("sampled FD receipt schema is invalid")
        if self.full_accumulator_snapshot_emitted is not False:
            raise ValueError(
                "sampled FD stream cannot emit a full accumulator snapshot"
            )
        if self.authority_scope != SAMPLED_DISCOVERY_AUTHORITY_SCOPE:
            raise ValueError("sampled FD receipt authority is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "route_windows", windows)
        object.__setattr__(self, "packet_boundary_index_before", before)
        object.__setattr__(self, "packet_boundary_index_after", after)
        object.__setattr__(self, "shared_tensor_observation_count", observations)

    def _hash_payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "request_hash",
                "design_hash",
                "append_result_hash",
                "route_axis_hash",
                "context_axis_hash",
                "stratum_axis_hash",
                "source_measurement_axis_hash",
                "destination_transport_hash",
                "raw_logit_stream_identity_hash",
                "device_identity",
                "memory_forecast_hash",
                "route_count",
                "registered_route_chunk_size",
                "route_windows",
                "packet_boundary_index_before",
                "packet_boundary_index_after",
                "shared_tensor_observation_count",
                "full_accumulator_snapshot_emitted",
                "authority_scope",
            )
        }
        if self.schema == SAMPLED_FD_STREAM_RECEIPT_SCHEMA_V2:
            payload["raw_target_logit_row_semantics"] = (
                self.raw_target_logit_row_semantics
            )
        return payload

    @property
    def receipt_hash(self) -> str:
        observed = content_hash(self._hash_payload(), prefix=_FD_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("sampled FD stream receipt was tampered")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


class _SampledStreamTransaction:
    """Shared fail-closed lifecycle for the two nominal adapters."""

    _failed: bool
    _failure_reason: str | None
    _packet_open: bool
    _closed: bool

    def _assert_live(self) -> None:
        if self._failed:
            raise RuntimeError(
                "sampled stream transaction is permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed:
            raise RuntimeError("sampled stream transaction is already closed")

    def _abort(self, error: Exception) -> None:
        self._failed = True
        self._failure_reason = f"{type(error).__name__}: {error}"
        raise error


class V1SampledReducedPacketLossOnlyAdapter(
    _SampledStreamTransaction,
    _RouteChunkedLossOnlyConsumer,
):
    """Nominal provider adapter for one GPU-reduced attribution packet."""

    positive_removal_stream_coordinate = RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE

    def __init__(
        self,
        request: V1SampledReducedPacketStreamRequest,
        *,
        dual_chunk_observer: Callable[[str, int, int, Any], None] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("sampled reduced streaming requires PyTorch")
        if type(request) is not V1SampledReducedPacketStreamRequest:
            raise TypeError("sampled reduced adapter requires its exact request")
        request.assert_integrity()
        if request.schema == SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2:
            if not callable(dual_chunk_observer):
                raise ValueError(
                    "canonical corrected-evidence streaming requires its shared "
                    "dual-chunk observer"
                )
        elif dual_chunk_observer is not None:
            raise ValueError(
                "legacy corrected-evidence streaming cannot claim a dual observer"
            )
        self.request = request
        self._dual_chunk_observer = dual_chunk_observer
        self.route_ids = request.route_ids
        self.device = _canonical_device(request.declared_device)
        self.loss_pullback_hash = request.raw_logit_stream_identity_hash
        self.destination_transport_hash = request.destination_transport_hash
        self.head_ids = ("positive_removal:raw_target_logit",)
        self.chunk_size = request.packet_request.registered_route_chunk_size
        self._consumer = V1GPUReducedProposalPacketConsumer(
            request.packet_request,
            device=self.device,
        )
        self._route_chunk_bound = False
        self._packet_open = False
        self._closed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._next_route = 0
        self._route_windows: list[tuple[int, int]] = []
        self._input_tensor_object_ids: list[int] = []
        self._closed_packet: V1GPUReducedProposalPacket | None = None
        self._receipt: V1SampledReducedStreamReceipt | None = None
        self._closure: V1SampledProposalDiscoveryClosure | None = None
        self._runtime_diagnostics: dict[str, float | int] = {
            "score_adapter_child_close_synchronization_inclusive_seconds": 0.0,
            "score_adapter_commit_construction_seconds": 0.0,
        }

    @property
    def input_tensor_object_ids(self) -> tuple[int, ...]:
        """Process-local telemetry proving a tee offered the same objects."""

        return tuple(self._input_tensor_object_ids)

    @property
    def runtime_diagnostics(self) -> Mapping[str, float | int]:
        """Return process-local metrics excluded from every typed hash."""

        return MappingProxyType(
            {
                **dict(self._consumer.runtime_diagnostics),
                **self._runtime_diagnostics,
            }
        )

    @property
    def closed_packet(self) -> V1GPUReducedProposalPacket:
        if self._closed_packet is None:
            raise RuntimeError("reduced packet is unavailable before full packet close")
        return self._closed_packet

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
            expected = self.request.packet_request.registered_route_chunk_size
            if observed != expected:
                raise ValueError("provider route chunk size differs from frozen packet")
            if self._packet_open:
                raise RuntimeError("cannot bind route chunks after packet open")
            self._consumer.request.assert_integrity(
                self.request.packet_request.request_hash
            )
            self._consumer.request.forecast_memory().assert_admissible()
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
            self.request.assert_integrity()
            if not self._route_chunk_bound:
                raise RuntimeError("provider route chunk size was not bound")
            if self._packet_open:
                raise RuntimeError("sampled reduced packet is already open")
            contexts = _identity_axis(
                context_ids,
                name="observed reduced packet contexts",
                unique=True,
            )
            if contexts != self.request.packet_request.expected_context_ids:
                raise ValueError("reduced packet contexts differ from frozen request")
            _assert_route_axis_matches(
                route_ids,
                self.route_ids,
                name="sampled reduced packet",
            )
            if str(destination_transport_hash or "") != self.destination_transport_hash:
                raise ValueError("reduced packet destination transport differs")
            if str(loss_pullback_hash or "") != self.loss_pullback_hash:
                raise ValueError("reduced packet raw-logit stream identity differs")
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
        begin = _nonnegative_int(start, name="route window start")
        end = _positive_int(stop, name="route window stop")
        route_count = self.request.packet_request.route_count
        width = end - begin
        if (
            not self._packet_open
            or begin != self._next_route
            or width <= 0
            or width > self.chunk_size
            or end > route_count
        ):
            raise ValueError(
                "reduced chunks must follow the registered exact route order"
            )
        if (
            not isinstance(rows, torch.Tensor)
            or tuple(rows.shape)
            != (self.request.packet_request.packet_draw_count, width)
            or rows.dtype != torch.float64
            or _canonical_device(rows.device) != self.device
            or not rows.is_contiguous()
        ):
            raise ValueError(
                "reduced chunk must be contiguous float64 [draw,route-window] "
                "on the declared device"
            )
        if not bool(torch.isfinite(rows).all().item()):
            raise ValueError("reduced chunk contains non-finite values")
        return begin, end

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        try:
            self._assert_live()
            begin, end = self._validate_positive_chunk(start, stop, rows)
            if self._dual_chunk_observer is not None:
                self._dual_chunk_observer(
                    "corrected_evidence",
                    begin,
                    end,
                    rows,
                )
            self._input_tensor_object_ids.append(id(rows))
            self._consumer.append_route_chunk(
                route_start=begin,
                signed_rows=rows,
                context_ids=self.request.packet_request.expected_context_ids,
                draw_ids=self.request.packet_request.expected_draw_ids,
                stratum_labels=self.request.packet_request.expected_stratum_labels,
            )
            self._route_windows.append((begin, end))
            self._next_route = end
        except Exception as error:
            self._abort(error)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._abort(
            PermissionError(
                "sampled reduced adapter refuses generic/BCE tripartite rows"
            )
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
                "sampled reduced adapter refuses consumer-side/fused pullback formation"
            )
        )

    def end_packet(self) -> None:
        try:
            self._assert_live()
            if (
                not self._packet_open
                or self._next_route != self.request.packet_request.route_count
            ):
                raise RuntimeError(
                    "reduced packet did not cover the route axis exactly"
                )
            child_close_started = time.perf_counter()
            packet = self._consumer.close()
            # Child close reaches the existing D2H barrier and is therefore
            # synchronization-inclusive without adding a diagnostic barrier.
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
                child_request_hash=packet.request.request_hash,
                child_packet_hash=packet.packet_hash,
                child_derivation_receipt_hash=(
                    packet.derivation_receipt.receipt_hash
                ),
                route_axis_hash=self.request.packet_request.route_axis_hash,
                context_axis_hash=content_hash(
                    self.request.packet_request.expected_context_ids,
                    prefix=_CONTEXT_AXIS_PREFIX,
                ),
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
                schema=(
                    SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA_V2
                    if self.request.schema
                    == SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2
                    else SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA
                ),
            )
            closure = V1SampledProposalDiscoveryClosure(
                adapter_kind="gpu_reduced_corrected_attribution_packet",
                adapter_request_hash=self.request.request_hash,
                adapter_receipt_hash=receipt.receipt_hash,
                child_contract_hash=packet.request.request_hash,
                child_result_hash=packet.packet_hash,
                child_result_schema=packet.schema,
                raw_logit_stream_identity_hash=self.loss_pullback_hash,
            )
            self._runtime_diagnostics[
                "score_adapter_commit_construction_seconds"
            ] = float(
                self._runtime_diagnostics[
                    "score_adapter_commit_construction_seconds"
                ]
            ) + (time.perf_counter() - commit_started)
            self._closed_packet = packet
            self._receipt = receipt
            self._closure = closure
            self._packet_open = False
            self._closed = True
        except Exception as error:
            self._abort(error)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        receipt = self.receipt
        closure = self.closure
        result = {
            "schema": receipt.schema,
            "typed_receipt_hash": receipt.receipt_hash,
            "typed_discovery_closure_hash": closure.closure_hash,
            "route_chunked_streaming": True,
            "route_chunk_size": receipt.registered_route_chunk_size,
            "route_chunk_update_count": len(receipt.route_windows),
            "loss_only_route_streaming": True,
            "sampled_reduced_packet_streaming": True,
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
            "proposal_discovery_input_authority": True,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }
        if self.request.schema == SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2:
            result["raw_target_logit_row_semantics"] = (
                self.request.raw_target_logit_row_semantics
            )
        return result


class V1SampledFDLossOnlyAdapter(
    _SampledStreamTransaction,
    _RouteChunkedLossOnlyConsumer,
):
    """Nominal same-stratum FD sink with one explicitly admitted row slab."""

    positive_removal_stream_coordinate = RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE

    def __init__(
        self,
        request: V1SampledFDStreamRequest,
        *,
        accumulator: V1StratifiedDeviceFDTrustAccumulator,
        dual_chunk_observer: Callable[[str, int, int, Any], None] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("sampled FD streaming requires PyTorch")
        if type(request) is not V1SampledFDStreamRequest:
            raise TypeError("sampled FD adapter requires its exact request")
        if type(accumulator) is not V1StratifiedDeviceFDTrustAccumulator:
            raise TypeError("sampled FD adapter requires its exact accumulator")
        request.assert_integrity()
        if request.schema == SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2:
            if not callable(dual_chunk_observer):
                raise ValueError(
                    "canonical literal-Jacobian streaming requires its shared "
                    "dual-chunk observer"
                )
        elif dual_chunk_observer is not None:
            raise ValueError(
                "legacy sampled FD streaming cannot claim a dual observer"
            )
        if accumulator.emit_derivation_receipt:
            raise ValueError(
                "sampled FD bridge requires the row-transfer-free accumulator mode; "
                "sealed source-measurement identities own this discovery closure"
            )
        if accumulator.design.design_hash != request.design.design_hash:
            raise ValueError("sampled FD accumulator design differs from request")
        device = _canonical_device(request.declared_device)
        if _canonical_device(accumulator.device) != device:
            raise ValueError("sampled FD accumulator device differs from request")
        boundary = _nonnegative_int(
            getattr(accumulator, "_packet_boundary_index"),
            name="accumulator packet boundary index",
        )
        if boundary != request.packet_index:
            raise ValueError("sampled FD request is stale at its packet boundary")
        request.memory_forecast.assert_admissible()

        self.request = request
        self._dual_chunk_observer = dual_chunk_observer
        self.accumulator = accumulator
        self.route_ids = request.design.route_ids
        self.device = device
        self.loss_pullback_hash = request.raw_logit_stream_identity_hash
        self.destination_transport_hash = request.destination_transport_hash
        self.head_ids = ("positive_removal:raw_target_logit",)
        self.chunk_size = request.registered_route_chunk_size
        self.memory_forecast = request.memory_forecast
        self._route_chunk_bound = False
        self._packet_open = False
        self._closed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._next_route = 0
        self._route_windows: list[tuple[int, int]] = []
        self._input_tensor_object_ids: list[int] = []
        self._packet_rows: Any | None = None
        self._append_result: V1SampledFDAppendResult | None = None
        self._receipt: V1SampledFDStreamReceipt | None = None
        self._closure: V1SampledProposalDiscoveryClosure | None = None
        self._runtime_diagnostics: dict[str, float | int] = {
            "fd_staging_allocation_host_wall_seconds": 0.0,
            "fd_staging_allocation_count": 0,
            "fd_staging_bytes": 0,
            "fd_staging_update_host_enqueue_seconds": 0.0,
            "fd_staging_update_count": 0,
            "fd_staging_update_bytes": 0,
            "fd_accumulator_update_synchronization_inclusive_seconds": 0.0,
            "fd_accumulator_update_count": 0,
            "fd_accumulator_update_input_bytes": 0,
            "fd_commit_construction_seconds": 0.0,
        }

    @property
    def input_tensor_object_ids(self) -> tuple[int, ...]:
        """Process-local telemetry proving a tee offered the same objects."""

        return tuple(self._input_tensor_object_ids)

    @property
    def runtime_diagnostics(self) -> Mapping[str, float | int]:
        """Return process-local metrics excluded from every typed hash."""

        return MappingProxyType(dict(self._runtime_diagnostics))

    @property
    def append_result(self) -> V1SampledFDAppendResult:
        if self._append_result is None:
            raise RuntimeError(
                "FD append result is unavailable before full packet close"
            )
        return self._append_result

    @property
    def receipt(self) -> V1SampledFDStreamReceipt:
        if self._receipt is None:
            raise RuntimeError("FD stream receipt is unavailable before close")
        return self._receipt

    @property
    def closure(self) -> V1SampledProposalDiscoveryClosure:
        if self._closure is None:
            raise RuntimeError("FD stream closure is unavailable before close")
        return self._closure

    def boundary_result(self) -> V1SampledFDAppendResult:
        """Return only a completed packet boundary; partial snapshots are refused."""

        if self._packet_open:
            raise RuntimeError("partial sampled FD packet snapshots are refused")
        return self.append_result

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        try:
            self._assert_live()
            observed = _positive_int(chunk_size, name="provider route chunk size")
            if observed != self.request.registered_route_chunk_size:
                raise ValueError(
                    "provider route chunk size differs from frozen FD packet"
                )
            if self._packet_open:
                raise RuntimeError("cannot bind route chunks after FD packet open")
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
            self.request.assert_integrity()
            if not self._route_chunk_bound:
                raise RuntimeError("provider route chunk size was not bound")
            if self._packet_open:
                raise RuntimeError("sampled FD packet is already open")
            contexts = _identity_axis(
                context_ids,
                name="observed FD packet contexts",
                unique=True,
            )
            if contexts != self.request.context_ids:
                raise ValueError("FD packet contexts differ from frozen request")
            _assert_route_axis_matches(
                route_ids,
                self.route_ids,
                name="sampled FD packet",
            )
            if str(destination_transport_hash or "") != self.destination_transport_hash:
                raise ValueError("FD packet destination transport differs")
            if str(loss_pullback_hash or "") != self.loss_pullback_hash:
                raise ValueError("FD packet raw-logit stream identity differs")
            boundary = _nonnegative_int(
                getattr(self.accumulator, "_packet_boundary_index"),
                name="accumulator packet boundary index",
            )
            if boundary != self.request.packet_index:
                raise ValueError("FD packet request became stale before allocation")
            if set(contexts).intersection(
                getattr(self.accumulator, "_all_context_ids")
            ):
                raise ValueError("FD packet contexts were already consumed")
            self.memory_forecast.assert_admissible()
            staging_started = time.perf_counter()
            rows = torch.empty(
                (len(contexts), len(self.route_ids)),
                dtype=torch.float64,
                device=self.device,
            )
            if int(rows.numel()) * int(rows.element_size()) != (
                self.memory_forecast.adapter_staging_device_bytes
            ):
                raise RuntimeError("FD staging allocation differs from memory forecast")
            self._runtime_diagnostics[
                "fd_staging_allocation_host_wall_seconds"
            ] = float(
                self._runtime_diagnostics["fd_staging_allocation_host_wall_seconds"]
            ) + (time.perf_counter() - staging_started)
            self._runtime_diagnostics["fd_staging_allocation_count"] = int(
                self._runtime_diagnostics["fd_staging_allocation_count"]
            ) + 1
            self._runtime_diagnostics["fd_staging_bytes"] = int(
                self._runtime_diagnostics["fd_staging_bytes"]
            ) + int(rows.numel()) * int(rows.element_size())
            self._packet_rows = rows
            self._next_route = 0
            self._packet_open = True
        except Exception as error:
            self._packet_rows = None
            self._abort(error)

    def _validate_positive_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> tuple[int, int]:
        begin = _nonnegative_int(start, name="route window start")
        end = _positive_int(stop, name="route window stop")
        route_count = len(self.route_ids)
        width = end - begin
        if (
            not self._packet_open
            or self._packet_rows is None
            or begin != self._next_route
            or width <= 0
            or width > self.chunk_size
            or end > route_count
        ):
            raise ValueError("FD chunks must follow the registered exact route order")
        if (
            not isinstance(rows, torch.Tensor)
            or tuple(rows.shape) != (len(self.request.context_ids), width)
            or rows.dtype != torch.float64
            or _canonical_device(rows.device) != self.device
            or not rows.is_contiguous()
        ):
            raise ValueError(
                "FD chunk must be contiguous float64 [draw,route-window] "
                "on the declared device"
            )
        if not bool(torch.isfinite(rows).all().item()):
            raise ValueError("FD chunk contains non-finite values")
        return begin, end

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        try:
            self._assert_live()
            begin, end = self._validate_positive_chunk(start, stop, rows)
            assert self._packet_rows is not None
            if self._dual_chunk_observer is not None:
                self._dual_chunk_observer(
                    "literal_jacobian",
                    begin,
                    end,
                    rows,
                )
            self._input_tensor_object_ids.append(id(rows))
            # The device-to-device copy is asynchronous.  Record only host
            # enqueue wall time and preserve the existing synchronization plan.
            staging_update_started = time.perf_counter()
            self._packet_rows[:, begin:end].copy_(rows.detach())
            self._runtime_diagnostics["fd_staging_update_host_enqueue_seconds"] = float(
                self._runtime_diagnostics["fd_staging_update_host_enqueue_seconds"]
            ) + (time.perf_counter() - staging_update_started)
            self._runtime_diagnostics["fd_staging_update_count"] = int(
                self._runtime_diagnostics["fd_staging_update_count"]
            ) + 1
            self._runtime_diagnostics["fd_staging_update_bytes"] = int(
                self._runtime_diagnostics["fd_staging_update_bytes"]
            ) + int(rows.numel()) * int(rows.element_size())
            self._route_windows.append((begin, end))
            self._next_route = end
        except Exception as error:
            self._abort(error)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._abort(
            PermissionError("sampled FD adapter refuses generic/BCE tripartite rows")
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
                "sampled FD adapter refuses consumer-side/fused pullback formation"
            )
        )

    def _child_ledger_hashes(self) -> tuple[str, str, int, tuple[str, ...]]:
        pairs = tuple(
            (
                name,
                tuple(self.accumulator._accumulators[name].context_ids),
            )
            for name in self.request.design.stratum_names
        )
        sources = tuple(
            (
                name,
                tuple(self.accumulator._source_measurements[name]),
            )
            for name in self.request.design.stratum_names
        )
        target = self.request.stratum_labels[0]
        child = self.accumulator._accumulators[target]
        return (
            content_hash(pairs, prefix=_CHILD_LEDGER_PREFIX),
            content_hash(sources, prefix=_SOURCE_AXIS_PREFIX),
            len(child.context_ids),
            tuple(child.last_update_source_row_hashes),
        )

    def end_packet(self) -> None:
        try:
            self._assert_live()
            if (
                not self._packet_open
                or self._packet_rows is None
                or self._next_route != len(self.route_ids)
            ):
                raise RuntimeError("FD packet did not cover the route axis exactly")
            before = _nonnegative_int(
                getattr(self.accumulator, "_packet_boundary_index"),
                name="accumulator packet boundary index",
            )
            if before != self.request.packet_index:
                raise ValueError("FD packet request became stale before commit")
            # The singleton response view allocates no second [draw, route] slab.
            update_started = time.perf_counter()
            self.accumulator.append_labeled_rows(
                rows=self._packet_rows[:, None, :],
                context_ids=self.request.context_ids,
                stratum_labels=self.request.stratum_labels,
                source_measurement_hashes=self.request.source_measurement_hashes,
            )
            # Accumulator validation reaches pre-existing scalar/D2H barriers,
            # so this wall-time span is deliberately synchronization-inclusive.
            self._runtime_diagnostics[
                "fd_accumulator_update_synchronization_inclusive_seconds"
            ] = float(
                self._runtime_diagnostics[
                    "fd_accumulator_update_synchronization_inclusive_seconds"
                ]
            ) + (time.perf_counter() - update_started)
            self._runtime_diagnostics["fd_accumulator_update_count"] = int(
                self._runtime_diagnostics["fd_accumulator_update_count"]
            ) + 1
            self._runtime_diagnostics["fd_accumulator_update_input_bytes"] = int(
                self._runtime_diagnostics["fd_accumulator_update_input_bytes"]
            ) + int(self._packet_rows.numel()) * int(self._packet_rows.element_size())
            after = _nonnegative_int(
                getattr(self.accumulator, "_packet_boundary_index"),
                name="accumulator packet boundary index",
            )
            commit_started = time.perf_counter()
            context_hash, source_hash, context_count, source_row_hashes = (
                self._child_ledger_hashes()
            )
            append_result = V1SampledFDAppendResult(
                request_hash=self.request.request_hash,
                design_hash=self.request.design.design_hash,
                stratum_name=self.request.stratum_labels[0],
                packet_boundary_index_before=before,
                packet_boundary_index_after=after,
                target_stratum_context_count_after=context_count,
                context_ledger_hash_after=context_hash,
                source_measurement_ledger_hash_after=source_hash,
                latest_source_row_hashes=source_row_hashes,
            )
            receipt = V1SampledFDStreamReceipt(
                request_hash=self.request.request_hash,
                design_hash=self.request.design.design_hash,
                append_result_hash=append_result.result_hash,
                route_axis_hash=self.request.design.route_axis_hash,
                context_axis_hash=content_hash(
                    self.request.context_ids,
                    prefix=_CONTEXT_AXIS_PREFIX,
                ),
                stratum_axis_hash=content_hash(
                    self.request.stratum_labels,
                    prefix=_CONTEXT_AXIS_PREFIX,
                ),
                source_measurement_axis_hash=content_hash(
                    self.request.source_measurement_hashes,
                    prefix=_SOURCE_AXIS_PREFIX,
                ),
                destination_transport_hash=self.destination_transport_hash,
                raw_logit_stream_identity_hash=self.loss_pullback_hash,
                device_identity=_device_identity(self.device),
                memory_forecast_hash=self.memory_forecast.forecast_hash,
                route_count=len(self.route_ids),
                registered_route_chunk_size=self.chunk_size,
                route_windows=tuple(self._route_windows),
                packet_boundary_index_before=before,
                packet_boundary_index_after=after,
                shared_tensor_observation_count=len(self._input_tensor_object_ids),
                raw_target_logit_row_semantics=(
                    self.request.raw_target_logit_row_semantics
                ),
                schema=(
                    SAMPLED_FD_STREAM_RECEIPT_SCHEMA_V2
                    if self.request.schema == SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2
                    else SAMPLED_FD_STREAM_RECEIPT_SCHEMA
                ),
            )
            closure = V1SampledProposalDiscoveryClosure(
                adapter_kind="same_stratum_fd_packet_append",
                adapter_request_hash=self.request.request_hash,
                adapter_receipt_hash=receipt.receipt_hash,
                child_contract_hash=self.request.design.design_hash,
                child_result_hash=append_result.result_hash,
                child_result_schema=append_result.schema,
                raw_logit_stream_identity_hash=self.loss_pullback_hash,
            )
            self._runtime_diagnostics["fd_commit_construction_seconds"] = float(
                self._runtime_diagnostics["fd_commit_construction_seconds"]
            ) + (time.perf_counter() - commit_started)
            self._append_result = append_result
            self._receipt = receipt
            self._closure = closure
            self._packet_rows = None
            self._packet_open = False
            self._closed = True
        except Exception as error:
            self._packet_rows = None
            self._abort(error)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        receipt = self.receipt
        closure = self.closure
        result = {
            "schema": receipt.schema,
            "typed_receipt_hash": receipt.receipt_hash,
            "typed_discovery_closure_hash": closure.closure_hash,
            "route_chunked_streaming": True,
            "route_chunk_size": receipt.registered_route_chunk_size,
            "route_chunk_update_count": len(receipt.route_windows),
            "loss_only_route_streaming": True,
            "sampled_same_stratum_fd_streaming": True,
            "positive_removal_stream_coordinate": (
                RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            ),
            "positive_removal_stream_identity_hash": self.loss_pullback_hash,
            "direct_positive_removal_chunk_count": len(receipt.route_windows),
            "loss_only_peak_staging_bytes": (
                self.memory_forecast.adapter_staging_device_bytes
            ),
            "packet_draw_joint_peak_device_bytes": (
                self.memory_forecast.joint_peak_device_bytes
            ),
            "packet_draw_memory_budget_bytes": (
                self.memory_forecast.packet_draw_memory_budget_bytes
            ),
            "full_tripartite_rows_materialized": False,
            "host_context_route_rows_transferred": False,
            "host_context_route_row_transfer_count": 0,
            "host_context_route_row_transfer_bytes": 0,
            "host_context_route_transfer_role": "none",
            "proposal_discovery_input_authority": True,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }
        if self.request.schema == SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2:
            result["raw_target_logit_row_semantics"] = (
                self.request.raw_target_logit_row_semantics
            )
            result["raw_target_logit_row_semantics_authority"] = (
                "typed_literal_gate_jacobian_input_only"
            )
        return result


__all__ = [
    "CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS",
    "LITERAL_GATE_JACOBIAN_ROW_SEMANTICS",
    "RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE",
    "SAMPLED_DISCOVERY_AUTHORITY_SCOPE",
    "SAMPLED_FD_STREAM_RECEIPT_SCHEMA_V2",
    "SAMPLED_FD_STREAM_REQUEST_SCHEMA_V2",
    "SAMPLED_REDUCED_STREAM_RECEIPT_SCHEMA_V2",
    "SAMPLED_REDUCED_STREAM_REQUEST_SCHEMA_V2",
    "V1SampledFDAppendResult",
    "V1SampledFDLossOnlyAdapter",
    "V1SampledFDPacketMemoryForecast",
    "V1SampledFDStreamReceipt",
    "V1SampledFDStreamRequest",
    "V1SampledProposalDiscoveryClosure",
    "V1SampledReducedPacketLossOnlyAdapter",
    "V1SampledReducedPacketStreamRequest",
    "V1SampledReducedStreamReceipt",
    "forecast_sampled_fd_packet_memory",
]
