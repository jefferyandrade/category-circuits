"""Typed bridge between the two historical sampled-V7 common-KKT names.

V7 D1 replay names the common relaxed coordinate
``spectral_plus_diagonal_l1_common_coordinate_v1``.  The residual packet
chain names the same physical two-branch shell source
``spectral_plus_diagonal_l1_provisional_shell_union_v1``.  Neither historical
identity is changed here.  This live receipt permits their use in one
full-population source only after replaying the exact D1 and residual owners
and proving their chart, mask, route, shell, native-fiber, source, resource,
evidence-stream, literal-J-stream, and alpha-provenance axes coincide.

The bridge grants coordinate-equivalence authority only.  It grants no F/Q,
sampling, stability, C/K/O, commit, terminal, performance, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_v7_d_fixed_source import (
    SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID,
    V1SampledV7D1FixedSourceResult,
)
from .modern_v1_sampled_variable_shell_packet_registration import (
    SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID,
)
from .modern_v1_signed_work_variable_shell_source_append import (
    V1SignedWorkVariableShellSourceAppendSegmentedHandle,
)


SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_common_KKT_coordinate_bridge_v1"
)
SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_AUTHORITY: Final = (
    "exact_live_V7_D1_common_coordinate_to_residual_provisional_shell_union_"
    "two_branch_coordinate_equivalence_only_no_F_Q_sampling_stability_C_K_O_"
    "commit_terminal_performance_or_pareto_authority_v1"
)
SAMPLED_V7_COMMON_KKT_CANONICAL_BRANCH_AXIS: Final = (
    "spectral",
    "diagonal_l1",
)

_PREFIX: Final = "partition-free-v1-sampled-v7-common-KKT-coordinate-bridge-"
_ISSUER: Final = object()


class V1SampledV7CommonKKTCoordinateBridgeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    HISTORICAL_ID_MISMATCH = "historical_id_mismatch"
    SOURCE_AXIS_MISMATCH = "source_axis_mismatch"
    COMPOSITION_MISMATCH = "composition_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7CommonKKTCoordinateBridgeRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7CommonKKTCoordinateBridgeRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CommonKKTCoordinateBridgeRefusalCode:
            raise TypeError("common-KKT bridge refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CommonKKTCoordinateBridgeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CommonKKTCoordinateBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CommonKKTCoordinateBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _live_axes(
    *,
    d1_fixed_source: V1SampledV7D1FixedSourceResult,
    residual_segmented_source: V1SignedWorkVariableShellSourceAppendSegmentedHandle,
):
    if (
        type(d1_fixed_source) is not V1SampledV7D1FixedSourceResult
        or type(residual_segmented_source)
        is not V1SignedWorkVariableShellSourceAppendSegmentedHandle
    ):
        _refuse(
            V1SampledV7CommonKKTCoordinateBridgeRefusalCode.INVALID_TYPE,
            "common-KKT bridge requires exact live D1 and residual owners",
        )
    try:
        d1_result_receipt = d1_fixed_source.receipt
        conditioned = d1_fixed_source.conditioned_d1_source
        conditioned.receipt.assert_integrity()
        d1 = conditioned.source_handle.receipt
        residual = residual_segmented_source.receipt
        projection = d1_fixed_source.literal_j_projection.registration
        projection.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7CommonKKTCoordinateBridgeRefusalCode.STALE_SOURCE,
            f"one common-KKT parent failed replay: {type(error).__name__}: {error}",
        )
    if (
        d1_result_receipt.conditioned_d1_source_receipt_hash
        != conditioned.receipt.receipt_hash
        or d1_result_receipt.conditioned_d1_compact_source_receipt_hash
        != d1.receipt_hash
        or projection.full_population_compact_source_receipt_hash != d1.receipt_hash
    ):
        _refuse(
            V1SampledV7CommonKKTCoordinateBridgeRefusalCode.STALE_SOURCE,
            "V7 fixed-source receipt does not retain its exact conditioned D1 owner",
        )
    if (
        d1.kkt_branch_id != SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID
        or residual.kkt_branch_id != SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID
    ):
        _refuse(
            V1SampledV7CommonKKTCoordinateBridgeRefusalCode.HISTORICAL_ID_MISMATCH,
            "parents do not name the two exact historical common-KKT identities",
        )
    shared = (
        (d1.model_identity_hash, residual.model_identity_hash),
        (d1.chart_identity_hash, residual.chart_identity_hash),
        (d1.current_mask_hash, residual.current_mask_hash),
        (d1.proposal_generation, residual.proposal_generation),
        (d1.route_axis_hash, residual.route_axis_hash),
        (d1.route_id_axis_hash, residual.route_id_axis_hash),
        (d1.route_count, residual.route_count),
        (d1.fiber_count, residual.fiber_count),
        (d1.cut_count, residual.cut_count),
        (d1.shell_freeze_receipt_hash, residual.shell_freeze_receipt_hash),
        (d1.shell_indices, residual.shell_indices),
        (d1.shell_route_ids, residual.shell_route_ids),
        (d1.shell_native_fiber_indices, residual.shell_native_fiber_indices),
        (
            d1.native_fiber_axis_receipt_hash,
            residual.native_fiber_axis_receipt_hash,
        ),
        (d1.native_fiber_measure_hash, residual.native_fiber_measure_hash),
        (
            d1.kkt_branch_source_receipt_hash,
            residual.kkt_branch_source_receipt_hash,
        ),
        (d1.resource_axis_hash, residual.resource_axis_hash),
        (d1.resource_source_receipt_hash, residual.resource_source_receipt_hash),
        (
            d1.corrected_evidence_stream_source_hash,
            residual.corrected_evidence_stream_source_hash,
        ),
        (d1.literal_j_stream_source_hash, residual.literal_j_stream_source_hash),
        (d1.alpha_ledger_id, residual.alpha_ledger_id),
        (d1.alpha_coordinate_id, residual.alpha_coordinate_id),
        (d1.alpha_lineage_hash, residual.alpha_lineage_hash),
    )
    if any(left != right for left, right in shared):
        _refuse(
            V1SampledV7CommonKKTCoordinateBridgeRefusalCode.SOURCE_AXIS_MISMATCH,
            "historical KKT names do not project the same live chart/shell/source axes",
        )
    if len(projection.direction_ids) != len(
        SAMPLED_V7_COMMON_KKT_CANONICAL_BRANCH_AXIS
    ):
        _refuse(
            V1SampledV7CommonKKTCoordinateBridgeRefusalCode.SOURCE_AXIS_MISMATCH,
            "V7 common coordinate does not retain exactly two private directions",
        )
    return d1_result_receipt, conditioned, d1, residual, projection


@dataclass(frozen=True, slots=True)
class V1SampledV7CommonKKTCoordinateBridgeReceipt:
    d1_fixed_source_receipt_hash: str
    conditioned_d1_source_receipt_hash: str
    conditioned_d1_device_source_receipt_hash: str
    residual_segmented_source_receipt_hash: str
    d1_historical_kkt_id: str
    residual_historical_kkt_id: str
    canonical_branch_axis: tuple[str, str]
    shared_kkt_source_receipt_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    shell_freeze_receipt_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    receipt_hash: str = field(init=False)
    exact_live_parent_replay: bool = True
    historical_ids_preserved: bool = True
    common_two_branch_coordinate_equivalence_authority: bool = True
    f_q_authority: bool = False
    statistical_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_commit_terminal_performance_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_AUTHORITY
    schema: str = SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_SCHEMA
    _d1_fixed_source: V1SampledV7D1FixedSourceResult = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _residual_segmented_source: (
        V1SignedWorkVariableShellSourceAppendSegmentedHandle
    ) = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _issuer_token: InitVar[object | None] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER:
            raise TypeError("common-KKT bridge receipts are issuer-only")
        for name in (
            "d1_fixed_source_receipt_hash",
            "conditioned_d1_source_receipt_hash",
            "conditioned_d1_device_source_receipt_hash",
            "residual_segmented_source_receipt_hash",
            "d1_historical_kkt_id",
            "residual_historical_kkt_id",
            "shared_kkt_source_receipt_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "authority_scope",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        required_true = (
            self.exact_live_parent_replay,
            self.historical_ids_preserved,
            self.common_two_branch_coordinate_equivalence_authority,
        )
        required_false = (
            self.f_q_authority,
            self.statistical_authority,
            self.d_stability_authority,
            self.c_k_o_commit_terminal_performance_or_pareto_authority,
        )
        if (
            self.d1_historical_kkt_id != SAMPLED_V7_COMMON_TWO_BRANCH_KKT_ID
            or self.residual_historical_kkt_id
            != SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID
            or self.canonical_branch_axis
            != SAMPLED_V7_COMMON_KKT_CANONICAL_BRANCH_AXIS
            or any(type(value) is not bool for value in required_true + required_false)
            or any(value is not True for value in required_true)
            or any(value is not False for value in required_false)
            or self.authority_scope
            != SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_AUTHORITY
            or self.schema != SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_SCHEMA
        ):
            _refuse(
                V1SampledV7CommonKKTCoordinateBridgeRefusalCode.AUTHORITY_VIOLATION,
                "common-KKT bridge crossed its coordinate-only authority",
            )
        observed = self._compute_hash()
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7CommonKKTCoordinateBridgeRefusalCode.STALE_SOURCE,
                "common-KKT bridge receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_") and name != "receipt_hash"
        }

    def _compute_hash(self) -> str:
        d1_result, conditioned, d1, residual, _projection = _live_axes(
            d1_fixed_source=self._d1_fixed_source,
            residual_segmented_source=self._residual_segmented_source,
        )
        observed = {
            "d1_fixed_source_receipt_hash": d1_result.receipt_hash,
            "conditioned_d1_source_receipt_hash": conditioned.receipt.receipt_hash,
            "conditioned_d1_device_source_receipt_hash": d1.receipt_hash,
            "residual_segmented_source_receipt_hash": residual.receipt_hash,
            "d1_historical_kkt_id": d1.kkt_branch_id,
            "residual_historical_kkt_id": residual.kkt_branch_id,
            "canonical_branch_axis": SAMPLED_V7_COMMON_KKT_CANONICAL_BRANCH_AXIS,
            "shared_kkt_source_receipt_hash": d1.kkt_branch_source_receipt_hash,
            "model_identity_hash": d1.model_identity_hash,
            "chart_identity_hash": d1.chart_identity_hash,
            "current_mask_hash": d1.current_mask_hash,
            "proposal_generation": d1.proposal_generation,
            "route_axis_hash": d1.route_axis_hash,
            "route_id_axis_hash": d1.route_id_axis_hash,
            "route_count": d1.route_count,
            "shell_freeze_receipt_hash": d1.shell_freeze_receipt_hash,
            "native_fiber_axis_receipt_hash": d1.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": d1.native_fiber_measure_hash,
            "corrected_evidence_stream_source_hash": d1.corrected_evidence_stream_source_hash,
            "literal_j_stream_source_hash": d1.literal_j_stream_source_hash,
            "resource_axis_hash": d1.resource_axis_hash,
            "resource_source_receipt_hash": d1.resource_source_receipt_hash,
            "alpha_ledger_id": d1.alpha_ledger_id,
            "alpha_coordinate_id": d1.alpha_coordinate_id,
            "alpha_lineage_hash": d1.alpha_lineage_hash,
        }
        if any(getattr(self, name) != value for name, value in observed.items()):
            _refuse(
                V1SampledV7CommonKKTCoordinateBridgeRefusalCode.STALE_SOURCE,
                "common-KKT bridge no longer matches its exact live parents",
            )
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_composition(self, *, conditioned_d1, residual_segmented_source) -> str:
        if (
            conditioned_d1 is not self._d1_fixed_source.conditioned_d1_source
            or residual_segmented_source is not self._residual_segmented_source
        ):
            _refuse(
                V1SampledV7CommonKKTCoordinateBridgeRefusalCode.COMPOSITION_MISMATCH,
                "bridge was presented with substituted D1 or residual custody",
            )
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._integrity_seal or observed != self.receipt_hash:
            _refuse(
                V1SampledV7CommonKKTCoordinateBridgeRefusalCode.STALE_SOURCE,
                "common-KKT bridge no longer replays",
            )
        return observed

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live common-KKT bridge receipts are nonserializable")


def issue_v1_sampled_v7_common_kkt_coordinate_bridge(
    *,
    d1_fixed_source: V1SampledV7D1FixedSourceResult,
    residual_segmented_source: V1SignedWorkVariableShellSourceAppendSegmentedHandle,
) -> V1SampledV7CommonKKTCoordinateBridgeReceipt:
    d1_result, conditioned, d1, residual, _projection = _live_axes(
        d1_fixed_source=d1_fixed_source,
        residual_segmented_source=residual_segmented_source,
    )
    return V1SampledV7CommonKKTCoordinateBridgeReceipt(
        d1_fixed_source_receipt_hash=d1_result.receipt_hash,
        conditioned_d1_source_receipt_hash=conditioned.receipt.receipt_hash,
        conditioned_d1_device_source_receipt_hash=d1.receipt_hash,
        residual_segmented_source_receipt_hash=residual.receipt_hash,
        d1_historical_kkt_id=d1.kkt_branch_id,
        residual_historical_kkt_id=residual.kkt_branch_id,
        canonical_branch_axis=SAMPLED_V7_COMMON_KKT_CANONICAL_BRANCH_AXIS,
        shared_kkt_source_receipt_hash=d1.kkt_branch_source_receipt_hash,
        model_identity_hash=d1.model_identity_hash,
        chart_identity_hash=d1.chart_identity_hash,
        current_mask_hash=d1.current_mask_hash,
        proposal_generation=d1.proposal_generation,
        route_axis_hash=d1.route_axis_hash,
        route_id_axis_hash=d1.route_id_axis_hash,
        route_count=d1.route_count,
        shell_freeze_receipt_hash=d1.shell_freeze_receipt_hash,
        native_fiber_axis_receipt_hash=d1.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=d1.native_fiber_measure_hash,
        corrected_evidence_stream_source_hash=d1.corrected_evidence_stream_source_hash,
        literal_j_stream_source_hash=d1.literal_j_stream_source_hash,
        resource_axis_hash=d1.resource_axis_hash,
        resource_source_receipt_hash=d1.resource_source_receipt_hash,
        alpha_ledger_id=d1.alpha_ledger_id,
        alpha_coordinate_id=d1.alpha_coordinate_id,
        alpha_lineage_hash=d1.alpha_lineage_hash,
        _d1_fixed_source=d1_fixed_source,
        _residual_segmented_source=residual_segmented_source,
        _issuer_token=_ISSUER,
    )


__all__ = (
    "SAMPLED_V7_COMMON_KKT_CANONICAL_BRANCH_AXIS",
    "SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_AUTHORITY",
    "SAMPLED_V7_COMMON_KKT_COORDINATE_BRIDGE_SCHEMA",
    "V1SampledV7CommonKKTCoordinateBridgeReceipt",
    "V1SampledV7CommonKKTCoordinateBridgeRefusal",
    "V1SampledV7CommonKKTCoordinateBridgeRefusalCode",
    "issue_v1_sampled_v7_common_kkt_coordinate_bridge",
)
