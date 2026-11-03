"""Source-derived D-uncertainty product for sampled-v7 signed-work pruning.

This module closes the narrow gap between the sampled full-population compact
source and the existing finite/continuum D-decision witnesses.  It does not
solve a KKT problem, register CUDA cells, measure C, calibrate, call the
paired oracle, commit a mask, or claim stability.  Its only job is to replace
test-authored ``V1SignedWorkDecisionUncertaintySet`` construction with a
receipt-derived owner product:

* signed corrected-F sampling from the compact source's corrected-evidence
  stream;
* literal-J sampling from the compact source's literal-J stream;
* deterministic FD/rank residual from an exact FD-geometry source; and
* deterministic solver/rounding uncertainty from the compact-dual solve
  receipt.

The finite realization axes intentionally contain only the registered current
source vertex unless an exact finite-population census is present.  Continuum
sampling/numerical/rank uncertainty is represented later by source-derived
owner radii and margin receipts, not by fabricated lower/upper grid points.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, Mapping, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_phase_a_global_q_compact_outer_lp_dual_solver import (
    V1PhaseAGlobalQCompactOuterLPDualDesign,
    V1PhaseAGlobalQCompactOuterLPDualSolveReceipt,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledFullPopulationCompactSourceReceipt,
)
from .modern_v1_sampled_device_final_rank_source import (
    V1SampledDeviceFinalRankSourceReceipt,
    V1SampledFDGeometrySourceCoordinate,
    V1SampledFDGeometrySourceProvenance,
    normalize_v1_sampled_fd_geometry_source,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJFDRankReplayReceipt,
)
from .modern_v1_sampled_v7_branch_decision_image import (
    v1_sampled_v7_branch_owner_source_derivation_hash,
)
from .modern_v1_signed_work_decision_enclosure import (
    V1SignedWorkDecisionOwnerRole,
    V1SignedWorkDecisionOwnerUncertaintyReceipt,
    derive_v1_signed_work_decision_owner_receipts,
)
from .modern_v1_signed_work_decision_witness import (
    V1SignedWorkDecisionUncertaintySet,
    V1SignedWorkFDResidualUncertaintySet,
    V1SignedWorkSamplingUncertaintyRole,
    V1SignedWorkSamplingUncertaintySet,
    V1SignedWorkSolverRoundingUncertaintySet,
)


SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_decision_uncertainty_source_v2"
)
SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_ALGORITHM: Final = (
    "compact_source_signed_F_literal_J_sampling_plus_exact_FD_geometry_source_"
    "and_compact_dual_solver_rounding_current_vertex_product_v2"
)
SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_AUTHORITY: Final = (
    "source_derived_D_uncertainty_product_only_no_D_stability_C_K_O_commit_"
    "terminal_performance_or_pareto_authority_v1"
)
SAMPLED_V7_D_DECISION_UNCERTAINTY_OWNER_RADIUS_RULE: Final = (
    "owner_radii_must_be_supplied_by_source_derived_statistical_FD_or_"
    "numerical_enclosures_before_branch_image_or_continuum_closure_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-D-decision-uncertainty-source-"
_ISSUER_TOKEN: Final = object()
_OWNER_ROLES: Final = (
    V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
    V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
    V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
    V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
)


class V1SampledV7DDecisionUncertaintySourceRefusalCode(str, Enum):
    """Typed fail-closed boundary for source-derived D uncertainty."""

    INVALID_TYPE = "invalid_type"
    SOURCE_MISMATCH = "source_mismatch"
    MIXED_PROVENANCE = "mixed_provenance"
    STALE_SOURCE = "stale_source"
    RANK_LADDER_MISMATCH = "rank_ladder_mismatch"
    SOLVER_ITERATION_MISMATCH = "solver_iteration_mismatch"
    REFERENCE_DOMAIN_EXCEEDED = "reference_domain_exceeded"
    MISSING_OWNER_RADIUS = "missing_owner_radius"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7DDecisionUncertaintySourceRefusal(RuntimeError):
    """Raised when sampled-v7 D uncertainty cannot be source-derived."""

    def __init__(
        self,
        code: V1SampledV7DDecisionUncertaintySourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DDecisionUncertaintySourceRefusalCode:
            raise TypeError("D uncertainty source refusal code has the wrong type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DDecisionUncertaintySourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DDecisionUncertaintySourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty tuple",
        )
    axis = tuple(_identity(value, name=f"{name} entry") for value in values)
    if len(set(axis)) != len(axis):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must not repeat entries",
        )
    return axis


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is bool or not isinstance(value, Integral):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be an integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is bool or not isinstance(value, Integral):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be an integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    if type(value) is bool or not isinstance(value, Real):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be a real number",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be finite and nonnegative",
        )
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact bool",
        )
    return value


def _rank_ladder(values: Sequence[int], *, current_rank: int) -> tuple[int, ...]:
    ladder = tuple(_positive_int(value, name="admitted FD rank") for value in values)
    if not ladder or ladder != tuple(sorted(set(ladder))) or current_rank not in ladder:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.RANK_LADDER_MISMATCH,
            "current FD rank must belong to one increasing admitted ladder",
        )
    return ladder


def _cuda_iteration_from_message(message: str) -> int | None:
    prefix = "cuda_pdhg_iteration_"
    if not message.startswith(prefix):
        return None
    try:
        return _positive_int(message[len(prefix) :], name="completed CUDA iteration")
    except V1SampledV7DDecisionUncertaintySourceRefusal:
        return None


@dataclass(frozen=True, slots=True)
class V1SampledV7DDecisionUncertaintySourceDesign:
    """Registered policy for constructing the current D uncertainty product."""

    admitted_rank_ladder: tuple[int, ...] = (8, 16, 32)
    signed_f_additional_registered_samples_available: bool = True
    literal_j_additional_registered_samples_available: bool = True
    maximum_reference_realizations: int = 4096
    solver_completed_iterations: int | None = None
    solver_maximum_registered_iterations: int | None = None

    def normalized(self, *, current_rank: int, finite_census: bool) -> dict[str, object]:
        rank = _positive_int(current_rank, name="current FD rank")
        return {
            "admitted_rank_ladder": _rank_ladder(
                self.admitted_rank_ladder,
                current_rank=rank,
            ),
            "signed_f_additional_registered_samples_available": (
                False
                if finite_census
                else _exact_bool(
                    self.signed_f_additional_registered_samples_available,
                    name="signed-F additional registered samples available",
                )
            ),
            "literal_j_additional_registered_samples_available": (
                False
                if finite_census
                else _exact_bool(
                    self.literal_j_additional_registered_samples_available,
                    name="literal-J additional registered samples available",
                )
            ),
            "maximum_reference_realizations": _positive_int(
                self.maximum_reference_realizations,
                name="maximum reference realizations",
            ),
            "solver_completed_iterations": (
                None
                if self.solver_completed_iterations is None
                else _nonnegative_int(
                    self.solver_completed_iterations,
                    name="solver completed iterations",
                )
            ),
            "solver_maximum_registered_iterations": (
                None
                if self.solver_maximum_registered_iterations is None
                else _positive_int(
                    self.solver_maximum_registered_iterations,
                    name="solver maximum registered iterations",
                )
            ),
        }

    @property
    def design_hash(self) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_sampled_v7_D_decision_uncertainty_source_design_v1",
                "admitted_rank_ladder": self.admitted_rank_ladder,
                "signed_f_additional_registered_samples_available": (
                    self.signed_f_additional_registered_samples_available
                ),
                "literal_j_additional_registered_samples_available": (
                    self.literal_j_additional_registered_samples_available
                ),
                "maximum_reference_realizations": self.maximum_reference_realizations,
                "solver_completed_iterations": self.solver_completed_iterations,
                "solver_maximum_registered_iterations": (
                    self.solver_maximum_registered_iterations
                ),
            },
            prefix=_PREFIX + "design-",
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7DDecisionUncertaintySourceReceipt:
    """Array-free receipt binding a compact source to one D uncertainty set."""

    design_hash: str
    source_receipt_hash: str
    source_replay_hash: str
    chart_identity_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_w_prefix_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hash: str
    alpha_lineage_hash: str
    finite_census_complete: bool
    signed_f_sampling_set_hash: str
    literal_j_sampling_set_hash: str
    fd_residual_set_hash: str
    solver_rounding_set_hash: str
    uncertainty_hash: str
    fd_geometry_source_provenance: V1SampledFDGeometrySourceProvenance
    fd_geometry_source_receipt_hash: str
    fd_geometry_source_coordinate_hash: str
    fd_rank_replay_receipt_hash: str | None
    fd_design_hash: str
    fd_snapshot_hash: str
    current_rank: int
    admitted_rank_ladder: tuple[int, ...]
    compact_dual_solve_receipt_hash: str
    solver_hash: str
    policy_hash: str
    solver_completed_iterations: int
    solver_maximum_registered_iterations: int
    maximum_reference_realizations: int
    one_current_vertex_per_uncertain_owner: bool = True
    owner_radii_not_inferred: bool = True
    source_derived_uncertainty_product: bool = True
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_ALGORITHM
    authority_scope: str = SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_AUTHORITY
    schema: str = SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.AUTHORITY_VIOLATION,
                "D uncertainty source receipts require their issuer",
            )
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            value = getattr(self, item.name)
            if item.name.endswith("_hash"):
                if item.name == "fd_rank_replay_receipt_hash" and value is None:
                    continue
                object.__setattr__(self, item.name, _identity(value, name=item.name))
        if type(self.fd_geometry_source_provenance) is not V1SampledFDGeometrySourceProvenance:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
                "FD geometry source provenance must be exact",
            )
        if self.fd_geometry_source_provenance is V1SampledFDGeometrySourceProvenance.HOST_CUSTODY_RANK_REPLAY:
            if self.fd_rank_replay_receipt_hash != self.fd_geometry_source_receipt_hash:
                _refuse(
                    V1SampledV7DDecisionUncertaintySourceRefusalCode.MIXED_PROVENANCE,
                    "host replay provenance must expose the legacy rank-replay hash",
                )
        elif self.fd_geometry_source_provenance is V1SampledFDGeometrySourceProvenance.DEVICE_RESIDENT_FINAL_RANK:
            if self.fd_rank_replay_receipt_hash is not None:
                _refuse(
                    V1SampledV7DDecisionUncertaintySourceRefusalCode.MIXED_PROVENANCE,
                    "device-resident FD geometry cannot claim a host rank replay hash",
                )
        else:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.MIXED_PROVENANCE,
                "unsupported FD geometry source provenance",
            )
        object.__setattr__(
            self,
            "registered_d_w_prefix_hash",
            _identity(
                self.registered_d_w_prefix_hash,
                name="registered_d_w_prefix_hash",
            ),
        )
        object.__setattr__(
            self,
            "admitted_rank_ladder",
            _rank_ladder(
                self.admitted_rank_ladder,
                current_rank=_positive_int(self.current_rank, name="current_rank"),
            ),
        )
        for name in (
            "solver_completed_iterations",
            "solver_maximum_registered_iterations",
            "maximum_reference_realizations",
        ):
            minimum = 0 if name == "solver_completed_iterations" else 1
            value = (
                _nonnegative_int(getattr(self, name), name=name)
                if minimum == 0
                else _positive_int(getattr(self, name), name=name)
            )
            object.__setattr__(self, name, value)
        if self.solver_completed_iterations > self.solver_maximum_registered_iterations:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.SOLVER_ITERATION_MISMATCH,
                "completed solver iterations exceed the registered maximum",
            )
        flags = (
            _exact_bool(
                self.finite_census_complete,
                name="finite_census_complete",
            ),
            _exact_bool(
                self.one_current_vertex_per_uncertain_owner,
                name="one_current_vertex_per_uncertain_owner",
            ),
            _exact_bool(self.owner_radii_not_inferred, name="owner_radii_not_inferred"),
            _exact_bool(
                self.source_derived_uncertainty_product,
                name="source_derived_uncertainty_product",
            ),
            _exact_bool(self.d_stability_authority, name="d_stability_authority"),
            _exact_bool(self.c_measurement_authority, name="c_measurement_authority"),
            _exact_bool(self.calibration_authority, name="calibration_authority"),
            _exact_bool(self.oracle_authority, name="oracle_authority"),
            _exact_bool(self.commit_authority, name="commit_authority"),
            _exact_bool(self.terminal_authority, name="terminal_authority"),
            _exact_bool(self.performance_authority, name="performance_authority"),
            _exact_bool(self.pareto_authority, name="pareto_authority"),
        )
        if flags[1:] != (True, True, True, False, False, False, False, False, False, False, False):
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.AUTHORITY_VIOLATION,
                "D uncertainty source receipt overclaimed authority",
            )
        if (
            self.algorithm != SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_ALGORITHM
            or self.authority_scope != SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_AUTHORITY
            or self.schema != SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_SCHEMA
        ):
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.AUTHORITY_VIOLATION,
                "unsupported D uncertainty source semantics",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.STALE_SOURCE,
                "D uncertainty source receipt no longer replays",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.STALE_SOURCE,
                "D uncertainty source receipt differs from expected hash",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7DDecisionUncertaintySourceResult:
    """Exact D uncertainty product plus its source-derived receipt."""

    receipt: V1SampledV7DDecisionUncertaintySourceReceipt
    uncertainty: V1SignedWorkDecisionUncertaintySet

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1SampledV7DDecisionUncertaintySourceReceipt
            or type(self.uncertainty) is not V1SignedWorkDecisionUncertaintySet
        ):
            raise TypeError("D uncertainty source result requires exact products")
        self.receipt.assert_integrity()
        if self.receipt.uncertainty_hash != self.uncertainty.uncertainty_hash:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.SOURCE_MISMATCH,
                "D uncertainty source receipt and uncertainty object diverged",
            )


def _sampling_realization_hash(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    role: V1SignedWorkSamplingUncertaintyRole,
) -> str:
    stream_hash = (
        source.corrected_evidence_stream_source_hash
        if role is V1SignedWorkSamplingUncertaintyRole.SIGNED_F
        else source.literal_j_stream_source_hash
    )
    if source.finite_census_complete:
        label = "exact-full-train-census"
    else:
        label = "registered-current-prefix"
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_sampling_realization_v1",
            "role": role.value,
            "label": label,
            "source_receipt_hash": source.receipt_hash,
            "source_replay_hash": source.source_replay_hash,
            "registered_d_prefix_hash": source.registered_d_prefix_hash,
            "context_axis_hash": source.context_axis_hash,
            "draw_axis_hash": source.draw_axis_hash,
            "stream_source_hash": stream_hash,
            "alpha_lineage_hash": source.alpha_lineage_hash,
            "finite_census_complete": source.finite_census_complete,
        },
        prefix=_PREFIX + "sampling-realization-",
    )


def _sampling_owner(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    role: V1SignedWorkSamplingUncertaintyRole,
    additional_registered_samples_available: bool,
) -> V1SignedWorkSamplingUncertaintySet:
    full_census = bool(source.finite_census_complete)
    stream_hash = (
        source.corrected_evidence_stream_source_hash
        if role is V1SignedWorkSamplingUncertaintyRole.SIGNED_F
        else source.literal_j_stream_source_hash
    )
    realization = _sampling_realization_hash(source=source, role=role)
    set_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_sampling_uncertainty_owner_v1",
            "role": role.value,
            "realization_hash": realization,
            "registered_d_prefix_hash": source.registered_d_prefix_hash,
            "context_axis_hash": source.context_axis_hash,
            "draw_axis_hash": source.draw_axis_hash,
            "direction_axis_hash": stream_hash,
            "alpha_lineage_hash": source.alpha_lineage_hash,
            "finite_census_complete": full_census,
        },
        prefix=_PREFIX + "sampling-owner-",
    )
    return V1SignedWorkSamplingUncertaintySet(
        role=role,
        uncertainty_set_hash=set_hash,
        realization_hashes=(realization,),
        registered_d_w_prefix_hash=source.registered_d_prefix_hash,
        context_axis_hash=source.context_axis_hash,
        draw_axis_hash=source.draw_axis_hash,
        direction_axis_hash=stream_hash,
        alpha_ledger_family_hash=source.alpha_lineage_hash,
        additional_registered_samples_available=(
            False if full_census else additional_registered_samples_available
        ),
        full_train_census=full_census,
        statistical_uncertainty=not full_census,
        deterministic_uncertainty=False,
        spends_alpha=not full_census,
    )


def _validate_fd_geometry_source(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    fd_geometry_source: (
        V1SampledLiteralJFDRankReplayReceipt | V1SampledDeviceFinalRankSourceReceipt
    ),
) -> V1SampledFDGeometrySourceCoordinate:
    try:
        coordinate = normalize_v1_sampled_fd_geometry_source(fd_geometry_source)
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.STALE_SOURCE,
            f"FD geometry source failed integrity: {type(error).__name__}: {error}",
        )
    if (
        coordinate.chart_identity_hash != source.chart_identity_hash
        or coordinate.current_mask_hash != source.current_mask_hash
        or coordinate.proposal_generation != source.proposal_generation
        or coordinate.route_axis_hash != source.route_axis_hash
        or coordinate.route_id_axis_hash != source.route_id_axis_hash
        or coordinate.route_count != source.route_count
        or coordinate.context_axis_hash != source.context_axis_hash
        or coordinate.draw_axis_hash != source.draw_axis_hash
        or coordinate.literal_j_stream_identity_hash
        != source.literal_j_stream_source_hash
        or coordinate.corrected_a_stream_identity_hash
        != source.corrected_evidence_stream_source_hash
        or (
            coordinate.registered_d_prefix_hash is not None
            and coordinate.registered_d_prefix_hash != source.registered_d_prefix_hash
        )
    ):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.SOURCE_MISMATCH,
            "FD geometry source is not bound to the compact D source",
        )
    return coordinate


def _validate_solver(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    compact_dual_solve_receipt: V1PhaseAGlobalQCompactOuterLPDualSolveReceipt,
    compact_dual_design: V1PhaseAGlobalQCompactOuterLPDualDesign | None,
    source_design: V1SampledV7DDecisionUncertaintySourceDesign,
) -> tuple[str, str, int, int]:
    try:
        solve_hash = compact_dual_solve_receipt.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.STALE_SOURCE,
            f"compact dual solve failed integrity: {type(error).__name__}: {error}",
        )
    if (
        compact_dual_solve_receipt.route_axis_hash != source.route_axis_hash
        or compact_dual_solve_receipt.current_mask_hash != source.current_mask_hash
        or compact_dual_solve_receipt.route_count != source.route_count
        or compact_dual_solve_receipt.d_stability_authority is not False
        or compact_dual_solve_receipt.c_measurement_authority is not False
        or compact_dual_solve_receipt.commit_authority is not False
    ):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.SOURCE_MISMATCH,
            "compact dual solve is not bound to the compact D source",
        )
    if (
        compact_dual_design is not None
        and compact_dual_design.design_hash != compact_dual_solve_receipt.design_hash
    ):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.SOLVER_ITERATION_MISMATCH,
            "compact dual design does not replay the solve receipt design hash",
        )
    completed = source_design.solver_completed_iterations
    if completed is None:
        completed = _cuda_iteration_from_message(compact_dual_solve_receipt.scipy_message)
    if completed is None:
        completed = 1
    completed = _nonnegative_int(completed, name="completed solver iterations")
    maximum = source_design.solver_maximum_registered_iterations
    if maximum is None:
        if (
            compact_dual_design is not None
            and compact_dual_solve_receipt.production_cuda_execution
        ):
            maximum = compact_dual_design.cuda_max_iterations
        else:
            maximum = completed
    maximum = _positive_int(maximum, name="maximum registered solver iterations")
    if completed > maximum:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.SOLVER_ITERATION_MISMATCH,
            "completed solver iterations exceed the registered maximum",
        )
    policy_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_solver_rounding_policy_v1",
            "compact_dual_design_hash": compact_dual_solve_receipt.design_hash,
            "feasibility_tolerance": compact_dual_solve_receipt.max_fractional_support_violation,
            "block_tolerance": compact_dual_solve_receipt.max_fractional_block_violation,
            "dual_constraint_tolerance": compact_dual_solve_receipt.max_dual_constraint_violation,
            "primal_dual_gap_upper": compact_dual_solve_receipt.primal_dual_gap_upper,
            "production_cuda_execution": compact_dual_solve_receipt.production_cuda_execution,
        },
        prefix=_PREFIX + "solver-policy-",
    )
    return solve_hash, policy_hash, completed, maximum


def _fd_residual_owner(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    fd_geometry_coordinate: V1SampledFDGeometrySourceCoordinate,
    admitted_rank_ladder: tuple[int, ...],
) -> V1SignedWorkFDResidualUncertaintySet:
    realization = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_FD_residual_realization_v1",
            "source_receipt_hash": source.receipt_hash,
            "fd_geometry_source_coordinate_hash": (
                fd_geometry_coordinate.coordinate_hash
            ),
            "fd_geometry_source_provenance": (
                fd_geometry_coordinate.provenance.value
            ),
            "fd_geometry_source_receipt_hash": (
                fd_geometry_coordinate.provenance_receipt_hash
            ),
            "fd_design_hash": fd_geometry_coordinate.fd_design_hash,
            "fd_snapshot_hash": fd_geometry_coordinate.fd_snapshot_hash,
            "same_capture_vjp_source_hash": source.same_capture_vjp_source_hash,
            "current_rank": fd_geometry_coordinate.requested_physical_rank,
            "admitted_rank_ladder": admitted_rank_ladder,
        },
        prefix=_PREFIX + "FD-realization-",
    )
    set_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_FD_residual_uncertainty_owner_v1",
            "realization_hash": realization,
            "registered_d_prefix_hash": source.registered_d_prefix_hash,
            "same_capture_vjp_source_hash": source.same_capture_vjp_source_hash,
            "current_rank": fd_geometry_coordinate.requested_physical_rank,
            "admitted_rank_ladder": admitted_rank_ladder,
        },
        prefix=_PREFIX + "FD-owner-",
    )
    return V1SignedWorkFDResidualUncertaintySet(
        uncertainty_set_hash=set_hash,
        realization_hashes=(realization,),
        registered_d_w_prefix_hash=source.registered_d_prefix_hash,
        same_capture_vjp_lineage_hash=source.same_capture_vjp_source_hash,
        current_rank=fd_geometry_coordinate.requested_physical_rank,
        admitted_rank_ladder=admitted_rank_ladder,
    )


def _solver_owner(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    solve_hash: str,
    policy_hash: str,
    compact_dual_solve_receipt: V1PhaseAGlobalQCompactOuterLPDualSolveReceipt,
    completed_iterations: int,
    maximum_registered_iterations: int,
) -> V1SignedWorkSolverRoundingUncertaintySet:
    realization = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_solver_rounding_realization_v1",
            "source_receipt_hash": source.receipt_hash,
            "compact_dual_solve_receipt_hash": solve_hash,
            "support_multiplier_digest": (
                compact_dual_solve_receipt.support_multiplier_digest
            ),
            "block_multiplier_digest": compact_dual_solve_receipt.block_multiplier_digest,
            "primal_fractional_witness_digest": (
                compact_dual_solve_receipt.primal_fractional_witness_digest
            ),
            "completed_iterations": completed_iterations,
            "maximum_registered_iterations": maximum_registered_iterations,
        },
        prefix=_PREFIX + "solver-realization-",
    )
    set_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_solver_rounding_uncertainty_owner_v1",
            "realization_hash": realization,
            "compact_dual_solve_receipt_hash": solve_hash,
            "policy_hash": policy_hash,
            "completed_iterations": completed_iterations,
            "maximum_registered_iterations": maximum_registered_iterations,
        },
        prefix=_PREFIX + "solver-owner-",
    )
    return V1SignedWorkSolverRoundingUncertaintySet(
        uncertainty_set_hash=set_hash,
        realization_hashes=(realization,),
        solver_hash=solve_hash,
        policy_hash=policy_hash,
        completed_iterations=completed_iterations,
        maximum_registered_iterations=maximum_registered_iterations,
    )


def issue_v1_sampled_v7_d_decision_uncertainty_from_compact_source(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    literal_j_rank_replay_receipt: V1SampledLiteralJFDRankReplayReceipt | None = None,
    fd_geometry_source: (
        V1SampledLiteralJFDRankReplayReceipt | V1SampledDeviceFinalRankSourceReceipt | None
    ) = None,
    compact_dual_solve_receipt: V1PhaseAGlobalQCompactOuterLPDualSolveReceipt,
    design: V1SampledV7DDecisionUncertaintySourceDesign = V1SampledV7DDecisionUncertaintySourceDesign(),
    compact_dual_design: V1PhaseAGlobalQCompactOuterLPDualDesign | None = None,
) -> V1SampledV7DDecisionUncertaintySourceResult:
    """Issue the source-derived current D uncertainty product.

    The returned object can feed finite-grid CUDA cell registration and, after
    separate source-derived owner radii are provided, branch-image owner
    receipt derivation.  It is not a decision-stability certificate.
    """

    if (
        type(source) is not V1SampledFullPopulationCompactSourceReceipt
        or type(compact_dual_solve_receipt)
        is not V1PhaseAGlobalQCompactOuterLPDualSolveReceipt
        or type(design) is not V1SampledV7DDecisionUncertaintySourceDesign
        or (
            compact_dual_design is not None
            and type(compact_dual_design) is not V1PhaseAGlobalQCompactOuterLPDualDesign
        )
    ):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            "D uncertainty source issuance requires exact typed receipts",
        )
    if (literal_j_rank_replay_receipt is None) == (fd_geometry_source is None):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            "D uncertainty source issuance requires exactly one FD geometry source",
        )
    if fd_geometry_source is None:
        fd_geometry_source = literal_j_rank_replay_receipt
    if type(fd_geometry_source) not in (
        V1SampledLiteralJFDRankReplayReceipt,
        V1SampledDeviceFinalRankSourceReceipt,
    ):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            "FD geometry source must be an exact host replay or device final-rank receipt",
        )
    try:
        source_hash = source.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.STALE_SOURCE,
            f"compact source failed integrity: {type(error).__name__}: {error}",
        )
    fd_coordinate = _validate_fd_geometry_source(
        source=source,
        fd_geometry_source=fd_geometry_source,
    )
    rank = fd_coordinate.requested_physical_rank
    normalized = design.normalized(
        current_rank=rank,
        finite_census=bool(source.finite_census_complete),
    )
    solve_hash, policy_hash, completed, maximum = _validate_solver(
        source=source,
        compact_dual_solve_receipt=compact_dual_solve_receipt,
        compact_dual_design=compact_dual_design,
        source_design=design,
    )
    signed_f = _sampling_owner(
        source=source,
        role=V1SignedWorkSamplingUncertaintyRole.SIGNED_F,
        additional_registered_samples_available=bool(
            normalized["signed_f_additional_registered_samples_available"]
        ),
    )
    literal_j = _sampling_owner(
        source=source,
        role=V1SignedWorkSamplingUncertaintyRole.LITERAL_J,
        additional_registered_samples_available=bool(
            normalized["literal_j_additional_registered_samples_available"]
        ),
    )
    fd_residual = _fd_residual_owner(
        source=source,
        fd_geometry_coordinate=fd_coordinate,
        admitted_rank_ladder=tuple(normalized["admitted_rank_ladder"]),
    )
    solver_rounding = _solver_owner(
        source=source,
        solve_hash=solve_hash,
        policy_hash=policy_hash,
        compact_dual_solve_receipt=compact_dual_solve_receipt,
        completed_iterations=completed,
        maximum_registered_iterations=maximum,
    )
    uncertainty = V1SignedWorkDecisionUncertaintySet(
        chart_identity_hash=source.chart_identity_hash,
        route_axis_hash=source.route_axis_hash,
        registered_d_w_prefix_hash=source.registered_d_prefix_hash,
        signed_f_sampling=signed_f,
        literal_j_sampling=literal_j,
        fd_residual=fd_residual,
        solver_rounding=solver_rounding,
        maximum_reference_realizations=int(normalized["maximum_reference_realizations"]),
    )
    if uncertainty.expected_realization_count > uncertainty.maximum_reference_realizations:
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.REFERENCE_DOMAIN_EXCEEDED,
            "registered D uncertainty product exceeds the reference-grid limit",
        )
    receipt = V1SampledV7DDecisionUncertaintySourceReceipt(
        design_hash=design.design_hash,
        source_receipt_hash=source_hash,
        source_replay_hash=source.source_replay_hash,
        chart_identity_hash=source.chart_identity_hash,
        route_axis_hash=source.route_axis_hash,
        route_id_axis_hash=source.route_id_axis_hash,
        registered_d_w_prefix_hash=source.registered_d_prefix_hash,
        context_axis_hash=source.context_axis_hash,
        draw_axis_hash=source.draw_axis_hash,
        corrected_evidence_stream_source_hash=(
            source.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=source.literal_j_stream_source_hash,
        same_capture_vjp_source_hash=source.same_capture_vjp_source_hash,
        alpha_lineage_hash=source.alpha_lineage_hash,
        finite_census_complete=source.finite_census_complete,
        signed_f_sampling_set_hash=signed_f.set_hash,
        literal_j_sampling_set_hash=literal_j.set_hash,
        fd_residual_set_hash=fd_residual.set_hash,
        solver_rounding_set_hash=solver_rounding.set_hash,
        uncertainty_hash=uncertainty.uncertainty_hash,
        fd_geometry_source_provenance=fd_coordinate.provenance,
        fd_geometry_source_receipt_hash=fd_coordinate.provenance_receipt_hash,
        fd_geometry_source_coordinate_hash=fd_coordinate.coordinate_hash,
        fd_rank_replay_receipt_hash=(
            fd_coordinate.provenance_receipt_hash
            if fd_coordinate.provenance
            is V1SampledFDGeometrySourceProvenance.HOST_CUSTODY_RANK_REPLAY
            else None
        ),
        fd_design_hash=fd_coordinate.fd_design_hash,
        fd_snapshot_hash=fd_coordinate.fd_snapshot_hash,
        current_rank=rank,
        admitted_rank_ladder=tuple(normalized["admitted_rank_ladder"]),
        compact_dual_solve_receipt_hash=solve_hash,
        solver_hash=solve_hash,
        policy_hash=policy_hash,
        solver_completed_iterations=completed,
        solver_maximum_registered_iterations=maximum,
        maximum_reference_realizations=int(normalized["maximum_reference_realizations"]),
        _issuance_token=_ISSUER_TOKEN,
    )
    return V1SampledV7DDecisionUncertaintySourceResult(
        receipt=receipt,
        uncertainty=uncertainty,
    )


def derive_v1_sampled_v7_branch_owner_receipts_from_uncertainty_source(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty_source: V1SampledV7DDecisionUncertaintySourceResult,
    branch: V1RelaxedKKTBranch,
    outward_radius_upper_bounds: Mapping[V1SignedWorkDecisionOwnerRole, float],
) -> tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]:
    """Derive branch-image owner receipts from source-bound uncertainty.

    Non-census sampling owners and deterministic FD/solver owners need
    explicit source-derived radii from their own statistical/numerical
    enclosures.  This function refuses missing radii rather than silently
    issuing zero-radius owner receipts.
    """

    if (
        type(source) is not V1SampledFullPopulationCompactSourceReceipt
        or type(uncertainty_source) is not V1SampledV7DDecisionUncertaintySourceResult
        or type(branch) is not V1RelaxedKKTBranch
    ):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.INVALID_TYPE,
            "branch owner derivation requires exact source, uncertainty, and branch",
        )
    if set(outward_radius_upper_bounds) != set(_OWNER_ROLES):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.MISSING_OWNER_RADIUS,
            "branch owner derivation requires explicit radii for all four owners",
        )
    uncertainty = uncertainty_source.uncertainty
    receipt = uncertainty_source.receipt
    receipt.assert_integrity()
    if (
        source.receipt_hash != receipt.source_receipt_hash
        or uncertainty.uncertainty_hash != receipt.uncertainty_hash
        or source.chart_identity_hash != uncertainty.chart_identity_hash
        or source.route_axis_hash != uncertainty.route_axis_hash
        or source.registered_d_prefix_hash != uncertainty.registered_d_w_prefix_hash
    ):
        _refuse(
            V1SampledV7DDecisionUncertaintySourceRefusalCode.SOURCE_MISMATCH,
            "branch owner derivation source differs from the uncertainty product",
        )
    radii = {
        role: _nonnegative_float(outward_radius_upper_bounds[role], name=role.value)
        for role in _OWNER_ROLES
    }
    for role, owner in (
        (V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING, uncertainty.signed_f_sampling),
        (V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING, uncertainty.literal_j_sampling),
    ):
        if owner.full_train_census and radii[role] != 0.0:
            _refuse(
                V1SampledV7DDecisionUncertaintySourceRefusalCode.MISSING_OWNER_RADIUS,
                "exact finite-population census owners must use zero radius",
            )
    derivations = {
        role: v1_sampled_v7_branch_owner_source_derivation_hash(
            source=source,
            uncertainty=uncertainty,
            branch=branch,
            owner_role=role,
        )
        for role in _OWNER_ROLES
    }
    return derive_v1_signed_work_decision_owner_receipts(
        uncertainty=uncertainty,
        source_derivation_hashes=derivations,
        outward_radius_upper_bounds=radii,
    )


__all__ = [
    "SAMPLED_V7_D_DECISION_UNCERTAINTY_OWNER_RADIUS_RULE",
    "SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_ALGORITHM",
    "SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_AUTHORITY",
    "SAMPLED_V7_D_DECISION_UNCERTAINTY_SOURCE_SCHEMA",
    "V1SampledV7DDecisionUncertaintySourceDesign",
    "V1SampledV7DDecisionUncertaintySourceReceipt",
    "V1SampledV7DDecisionUncertaintySourceRefusal",
    "V1SampledV7DDecisionUncertaintySourceRefusalCode",
    "V1SampledV7DDecisionUncertaintySourceResult",
    "derive_v1_sampled_v7_branch_owner_receipts_from_uncertainty_source",
    "issue_v1_sampled_v7_d_decision_uncertainty_from_compact_source",
]
