"""Typed proposal provenance for sampled pre-freeze response geometry.

The sampled modern-v1 path retains route-resolved sufficient statistics and
stratified Frequent-Directions snapshots, not the complete ordered raw-row
ledger required by :class:`GlobalTrustGeometryDerivationReceipt`.  This module
therefore owns a distinct certificate.  It proves exact numerical replay from
the admitted sampled snapshot into selector coordinates while explicitly
refusing source-row, population-geometry, F/Q, oracle, mask-commit, and
terminal authority.

The certificate factory also materializes each selector-scale route array
exactly once.  The production selector can then admit those immutable arrays
without another full-axis copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .contracts import content_hash
from .modern_v1_geometric_budgets import (
    V1GeometricTrustBudgets,
    same_geometric_coordinate,
)
from .modern_v1_neutral_runtime import DeclaredRouteAxis, FixedAffineBCECalibration
from .modern_v1_sampled_acquisition_adapters import (
    V1SampledPreFreezeGeometrySource,
)
from .modern_v1_sampled_proposal_streaming import (
    RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
)


SAMPLED_SNAPSHOT_SELECTOR_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_sampled_snapshot_selector_certificate_v1"
)
SAMPLED_SELECTOR_CHART_COORDINATES_SCHEMA: Final = (
    "partition_free_v1_sampled_selector_chart_coordinates_v1"
)
SAMPLED_SCORE_DISCOVERY_COORDINATES_SCHEMA: Final = (
    "partition_free_v1_sampled_score_discovery_coordinates_v1"
)
SAMPLED_SELECTOR_CHART_PROVENANCE_SCHEMA: Final = (
    "partition_free_v1_sampled_selector_chart_provenance_v1"
)
SAMPLED_SNAPSHOT_SELECTOR_NUMERICS_SCHEMA: Final = (
    "partition_free_v1_sampled_snapshot_selector_numerics_v1"
)
SAMPLED_PROPOSAL_POSITION_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_sampled_proposal_position_certificate_v1"
)

SAMPLED_SNAPSHOT_PROVENANCE_SCOPE: Final = (
    "sampled_snapshot_numerical_replay_for_proposal_composition_only_"
    "no_source_row_population_geometry_fq_bce_mask_or_terminal_authority_v1"
)
SAMPLED_CORRECTED_POINT_STATUS: Final = (
    "observed_sealed_prefix_class_balanced_point_estimate_no_census_or_"
    "population_exactness_authority_v1"
)
SAMPLED_PROPOSAL_POSITION_AUTHORITY_SCOPE: Final = (
    "sampled_selector_proposal_position_correspondence_only_no_irreversible_"
    "application_fq_bce_mask_commit_production_or_terminal_authority_v1"
)
SAMPLED_PROPOSAL_POSITION_RULE: Final = (
    "ordered_route_id_position_i_corresponds_to_physical_mask_bit_i"
)

SAMPLED_CORRECTED_PATH_HASH_PREFIX: Final = (
    "partition-free-v1-sampled-selector-corrected-path-"
)
SAMPLED_SIGNED_COVECTOR_HASH_PREFIX: Final = (
    "partition-free-v1-sampled-selector-signed-covector-"
)
SAMPLED_LOW_RANK_FACTOR_HASH_PREFIX: Final = (
    "partition-free-v1-sampled-selector-low-rank-factor-"
)
SAMPLED_RESIDUAL_DIAGONAL_HASH_PREFIX: Final = (
    "partition-free-v1-sampled-selector-residual-diagonal-"
)

_CERTIFICATE_PREFIX: Final = "partition-free-v1-sampled-snapshot-selector-certificate-"
_CORRECTED_POINT_PREFIX: Final = "partition-free-v1-sampled-selector-corrected-point-"
_COORDINATES_PREFIX: Final = "partition-free-v1-sampled-selector-chart-coordinates-"
_DISCOVERY_PREFIX: Final = "partition-free-v1-sampled-score-discovery-coordinates-"
_PROVENANCE_PREFIX: Final = "partition-free-v1-sampled-selector-chart-provenance-"
_NUMERICS_PREFIX: Final = "partition-free-v1-sampled-snapshot-selector-numerics-"
_POSITION_CERTIFICATE_PREFIX: Final = (
    "partition-free-v1-sampled-proposal-position-certificate-"
)
_ISSUANCE_TOKEN = object()
_NUMERICS_ISSUANCE_TOKEN = object()
_POSITION_ISSUANCE_TOKEN = object()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _readonly_f64_vector(
    value: object,
    *,
    name: str,
    size: int,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float64)
        or array.ndim != 1
        or array.shape != (size,)
        or not array.flags.c_contiguous
        or array.flags.writeable
        or not np.isfinite(array).all()
        or (nonnegative and bool(np.any(array < 0.0)))
    ):
        raise ValueError(
            f"{name} must be one immutable C-contiguous finite float64 vector"
        )
    return array


def _readonly_f64_matrix(
    value: object,
    *,
    name: str,
    columns: int,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float64)
        or array.ndim != 2
        or array.shape[1] != columns
        or not array.flags.c_contiguous
        or array.flags.writeable
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must be one immutable C-contiguous finite float64 matrix"
        )
    return array


def _immutable_f64_matrix(value: object, *, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite route-aligned matrix")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    # The caller has already materialized the combined mean-plus-FD factor.
    # Freezing that owned contiguous allocation in place avoids a second
    # route-scale byte copy (several GiB on the AlexNet route axis).
    contiguous.setflags(write=False)
    return contiguous


def _resource_cost_hash(
    *,
    route_count: int,
    uniform_resource_cost: float | None,
    resource_cost: np.ndarray | None,
) -> str:
    return content_hash(
        {
            "route_count": route_count,
            "uniform_resource_cost": uniform_resource_cost,
            "resource_cost": resource_cost,
        },
        prefix="partition-free-v1-production-selector-resource-cost-",
    )


@dataclass(frozen=True, slots=True, init=False)
class V1SampledSnapshotSelectorCertificate:
    """Factory-issued replay certificate for one sampled selector chart.

    The certificate closes the retained sufficient-state graph and exact
    selector numerics.  Its FD residual statement is conditional on the
    admitted sampled snapshot; it is neither a raw-row derivation proof nor a
    confidence statement about the finite-population response geometry.
    """

    sampled_geometry_source_hash: str
    admission_receipt_hash: str
    assembly_receipt_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    model_identity_hash: str
    query_id: str
    query_contract_hash: str
    preprocessing_hash: str
    sampling_law_hash: str
    current_mask_hash: str
    proposal_generation: int
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    sampled_look_index: int
    calibration_identity: str
    raw_logit_stream_source_hash: str
    raw_logit_coordinate: str
    context_mixture_hash: str
    sample_weight_hash: str
    source_measurement_axis_hash: str
    corrected_moments_state_hash: str
    corrected_path_cost_hash: str
    signed_covector_hash: str
    corrected_point_receipt_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    snapshot_numerical_backend_hash: str
    geometry_state_hash: str
    geometry_result_hash: str
    numerical_assembly_receipt_hash: str
    mean_component_hash: str
    centered_factor_hash: str
    residual_diagonal_hash: str
    low_rank_factor_hash: str
    mean_row_count: int
    centered_rank: int
    residual_spectral_bound: float
    raw_row_derivation_absence_hash: str
    sampled_snapshot_numerical_replay_authority: bool = True
    fd_numerical_residual_replay_authority: bool = True
    raw_row_derivation_receipt_present: bool = False
    source_row_proof_authority: bool = False
    finite_population_geometry_uncertainty_certified: bool = False
    sampled_geometry_statistical_optimality_claim: bool = False
    proposal_only: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = SAMPLED_SNAPSHOT_PROVENANCE_SCOPE
    schema: str = SAMPLED_SNAPSHOT_SELECTOR_CERTIFICATE_SCHEMA

    def __init__(
        self,
        *,
        _issuance_token: object | None = None,
        **values: object,
    ) -> None:
        if _issuance_token is not _ISSUANCE_TOKEN:
            raise TypeError(
                "sampled snapshot selector certificates can only be issued from "
                "an admitted V1SampledPreFreezeGeometrySource"
            )
        for item in self.__dataclass_fields__.values():
            if not item.init:
                continue
            value = values.pop(item.name, item.default)
            object.__setattr__(self, item.name, value)
        if values:
            raise TypeError(
                "unexpected sampled snapshot certificate fields: "
                f"{tuple(sorted(values))}"
            )
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "sampled_geometry_source_hash",
            "admission_receipt_hash",
            "assembly_receipt_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "model_identity_hash",
            "query_id",
            "query_contract_hash",
            "preprocessing_hash",
            "sampling_law_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "calibration_identity",
            "raw_logit_stream_source_hash",
            "context_mixture_hash",
            "sample_weight_hash",
            "source_measurement_axis_hash",
            "corrected_moments_state_hash",
            "corrected_path_cost_hash",
            "signed_covector_hash",
            "corrected_point_receipt_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "snapshot_numerical_backend_hash",
            "geometry_state_hash",
            "geometry_result_hash",
            "numerical_assembly_receipt_hash",
            "mean_component_hash",
            "centered_factor_hash",
            "residual_diagonal_hash",
            "low_rank_factor_hash",
            "raw_row_derivation_absence_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        for name in ("route_count", "sampled_look_index", "mean_row_count"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "centered_rank",
            _nonnegative_int(self.centered_rank, name="centered_rank"),
        )
        object.__setattr__(
            self,
            "residual_spectral_bound",
            _finite_nonnegative(
                self.residual_spectral_bound,
                name="residual spectral bound",
            ),
        )
        if self.physical_route_axis_hash == self.ordered_route_axis_hash:
            raise ValueError(
                "sampled selector physical and ordered route axes must remain distinct"
            )
        if self.raw_logit_coordinate != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
            raise ValueError("sampled selector certificate changed the raw-logit coordinate")
        required_true = (
            self.sampled_snapshot_numerical_replay_authority,
            self.fd_numerical_residual_replay_authority,
            self.proposal_only,
        )
        required_false = (
            self.raw_row_derivation_receipt_present,
            self.source_row_proof_authority,
            self.finite_population_geometry_uncertainty_certified,
            self.sampled_geometry_statistical_optimality_claim,
            self.fq_certification_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in (*required_true, *required_false)):
            raise TypeError("sampled selector authority fields must be exact booleans")
        if not all(required_true) or any(required_false):
            raise PermissionError(
                "sampled snapshot selector certificate crossed its proposal boundary"
            )
        if (
            self.authority_scope != SAMPLED_SNAPSHOT_PROVENANCE_SCOPE
            or self.schema != SAMPLED_SNAPSHOT_SELECTOR_CERTIFICATE_SCHEMA
        ):
            raise ValueError("unsupported sampled snapshot selector certificate scope")

    @property
    def certificate_hash(self) -> str:
        return content_hash(self, prefix=_CERTIFICATE_PREFIX)

    def assert_selector_payload(
        self,
        *,
        corrected_path_cost: np.ndarray,
        signed_covector: np.ndarray | None,
        low_rank_factor: np.ndarray,
        residual_diagonal: np.ndarray,
        residual_spectral_bound: float,
    ) -> None:
        """Recompute every route-array identity admitted to the selector."""

        if signed_covector is None:
            raise ValueError("sampled selector charts require their signed covector")
        observed = (
            (
                content_hash(
                    corrected_path_cost,
                    prefix=SAMPLED_CORRECTED_PATH_HASH_PREFIX,
                ),
                self.corrected_path_cost_hash,
                "corrected path",
            ),
            (
                content_hash(
                    signed_covector,
                    prefix=SAMPLED_SIGNED_COVECTOR_HASH_PREFIX,
                ),
                self.signed_covector_hash,
                "signed covector",
            ),
            (
                content_hash(
                    low_rank_factor,
                    prefix=SAMPLED_LOW_RANK_FACTOR_HASH_PREFIX,
                ),
                self.low_rank_factor_hash,
                "low-rank factor",
            ),
            (
                content_hash(
                    residual_diagonal,
                    prefix=SAMPLED_RESIDUAL_DIAGONAL_HASH_PREFIX,
                ),
                self.residual_diagonal_hash,
                "residual diagonal",
            ),
        )
        for actual, expected, label in observed:
            if actual != expected:
                raise ValueError(f"sampled selector {label} differs from its certificate")
        if low_rank_factor.shape != (
            self.mean_row_count + self.centered_rank,
            self.route_count,
        ):
            raise ValueError("sampled selector factor row decomposition differs")
        if not same_geometric_coordinate(
            residual_spectral_bound,
            self.residual_spectral_bound,
        ):
            raise ValueError("sampled selector residual spectral bound differs")


@dataclass(frozen=True, slots=True, init=False)
class V1SampledSnapshotSelectorNumerics:
    """Immutable full-axis arrays and their factory-issued certificate."""

    certificate: V1SampledSnapshotSelectorCertificate
    corrected_path_cost: np.ndarray = field(repr=False, compare=False)
    signed_covector: np.ndarray = field(repr=False, compare=False)
    low_rank_factor: np.ndarray = field(repr=False, compare=False)
    residual_diagonal: np.ndarray = field(repr=False, compare=False)
    schema: str = SAMPLED_SNAPSHOT_SELECTOR_NUMERICS_SCHEMA

    def __init__(
        self,
        *,
        certificate: V1SampledSnapshotSelectorCertificate,
        corrected_path_cost: np.ndarray,
        signed_covector: np.ndarray,
        low_rank_factor: np.ndarray,
        residual_diagonal: np.ndarray,
        schema: str = SAMPLED_SNAPSHOT_SELECTOR_NUMERICS_SCHEMA,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _NUMERICS_ISSUANCE_TOKEN:
            raise TypeError(
                "sampled snapshot selector numerics can only be issued by the "
                "pre-freeze source factory"
            )
        object.__setattr__(self, "certificate", certificate)
        object.__setattr__(self, "corrected_path_cost", corrected_path_cost)
        object.__setattr__(self, "signed_covector", signed_covector)
        object.__setattr__(self, "low_rank_factor", low_rank_factor)
        object.__setattr__(self, "residual_diagonal", residual_diagonal)
        object.__setattr__(self, "schema", schema)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.certificate) is not V1SampledSnapshotSelectorCertificate:
            raise TypeError("sampled selector numerics require their exact certificate")
        count = self.certificate.route_count
        path = _readonly_f64_vector(
            self.corrected_path_cost,
            name="sampled corrected path",
            size=count,
            nonnegative=True,
        )
        signed = _readonly_f64_vector(
            self.signed_covector,
            name="sampled signed covector",
            size=count,
        )
        factor = _readonly_f64_matrix(
            self.low_rank_factor,
            name="sampled low-rank factor",
            columns=count,
        )
        residual = _readonly_f64_vector(
            self.residual_diagonal,
            name="sampled residual diagonal",
            size=count,
            nonnegative=True,
        )
        self.certificate.assert_selector_payload(
            corrected_path_cost=path,
            signed_covector=signed,
            low_rank_factor=factor,
            residual_diagonal=residual,
            residual_spectral_bound=self.certificate.residual_spectral_bound,
        )
        if self.schema != SAMPLED_SNAPSHOT_SELECTOR_NUMERICS_SCHEMA:
            raise ValueError("unsupported sampled snapshot selector numerics schema")
        object.__setattr__(self, "corrected_path_cost", path)
        object.__setattr__(self, "signed_covector", signed)
        object.__setattr__(self, "low_rank_factor", factor)
        object.__setattr__(self, "residual_diagonal", residual)

    @property
    def numerics_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "certificate_hash": self.certificate.certificate_hash,
                "corrected_path_cost_hash": self.certificate.corrected_path_cost_hash,
                "signed_covector_hash": self.certificate.signed_covector_hash,
                "low_rank_factor_hash": self.certificate.low_rank_factor_hash,
                "residual_diagonal_hash": self.certificate.residual_diagonal_hash,
            },
            prefix=_NUMERICS_PREFIX,
        )


def prepare_v1_sampled_snapshot_selector_numerics(
    source: V1SampledPreFreezeGeometrySource,
) -> V1SampledSnapshotSelectorNumerics:
    """Issue truthful selector numerics from one exact pre-freeze source."""

    if type(source) is not V1SampledPreFreezeGeometrySource:
        raise TypeError("sampled selector preparation requires its exact source type")
    admission = source.admission
    admission_receipt = admission.receipt
    assembly_receipt = source.receipt
    geometry_result = source.geometry_result
    geometry = source.geometry_state
    numerical_receipt = geometry_result.receipt
    moments = admission.moments
    source_graph_pairs = (
        (assembly_receipt.admission_receipt_hash, admission_receipt.receipt_hash),
        (assembly_receipt.admission_hash, admission.admission_hash),
        (assembly_receipt.campaign_contract_hash, admission_receipt.campaign_contract_hash),
        (assembly_receipt.campaign_binding_hash, admission_receipt.campaign_binding_hash),
        (assembly_receipt.current_mask_hash, admission_receipt.current_mask_hash),
        (
            assembly_receipt.physical_route_axis_hash,
            admission_receipt.physical_route_axis_hash,
        ),
        (
            assembly_receipt.ordered_route_axis_hash,
            admission_receipt.ordered_route_axis_hash,
        ),
        (assembly_receipt.proposal_generation, admission_receipt.proposal_generation),
        (assembly_receipt.calibration_identity, admission_receipt.calibration_identity),
        (assembly_receipt.sample_weight_hash, admission_receipt.sample_weight_hash),
        (
            assembly_receipt.corrected_moments_state_hash,
            admission_receipt.corrected_moments_state_hash,
        ),
        (assembly_receipt.fd_design_hash, admission_receipt.fd_design_hash),
        (assembly_receipt.fd_snapshot_hash, admission_receipt.fd_snapshot_hash),
        (assembly_receipt.geometry_state_hash, geometry.content_hash),
        (assembly_receipt.geometry_result_hash, geometry_result.result_hash),
        (moments.state_hash, admission_receipt.corrected_moments_state_hash),
    )
    if (
        geometry.derivation_receipt is not None
        or assembly_receipt.global_trust_derivation_receipt_present is not False
        or assembly_receipt.raw_context_route_rows_retained is not False
        or assembly_receipt.full_per_draw_source_row_hash_ledger_retained is not False
        or geometry.residual_diagonal_bound is None
        or numerical_receipt.combined_geometry_hash != geometry.content_hash
        or numerical_receipt.design_hash != admission_receipt.fd_design_hash
        or numerical_receipt.receipt_hash
        != assembly_receipt.numerical_assembly_receipt_hash
        or source.geometry_result.result_hash != assembly_receipt.geometry_result_hash
        or any(left != right for left, right in source_graph_pairs)
    ):
        raise ValueError("sampled selector source graph does not close truthfully")
    admission.fd_design.assert_integrity(admission_receipt.fd_design_hash)
    if admission.fd_snapshot.assert_integrity() != admission_receipt.fd_snapshot_hash:
        raise ValueError("sampled selector FD snapshot differs from its admission")

    path = admission.absolute_point
    signed = admission.signed_point
    mean = np.asarray(geometry.mean_component, dtype=np.float64)
    centered = np.asarray(geometry.centered_factor, dtype=np.float64)
    factor = _immutable_f64_matrix(
        np.concatenate((mean, centered), axis=0),
        name="sampled mean-plus-centered factor",
        columns=admission_receipt.route_count,
    )
    residual_source = np.asarray(geometry.residual_diagonal_bound)
    if residual_source.flags.c_contiguous and not residual_source.flags.writeable:
        residual = residual_source
    else:
        residual = np.frombuffer(
            np.ascontiguousarray(residual_source, dtype=np.float64).tobytes(order="C"),
            dtype=np.float64,
        )
    path_hash = content_hash(path, prefix=SAMPLED_CORRECTED_PATH_HASH_PREFIX)
    signed_hash = content_hash(signed, prefix=SAMPLED_SIGNED_COVECTOR_HASH_PREFIX)
    factor_hash = content_hash(factor, prefix=SAMPLED_LOW_RANK_FACTOR_HASH_PREFIX)
    residual_hash = content_hash(
        residual,
        prefix=SAMPLED_RESIDUAL_DIAGONAL_HASH_PREFIX,
    )
    if (
        numerical_receipt.mean_component_hash
        != content_hash(mean, prefix="partition-free-v1-stratified-fd-mean-")
        or numerical_receipt.centered_factor_hash
        != content_hash(centered, prefix="partition-free-v1-stratified-fd-factor-")
        or numerical_receipt.residual_diagonal_hash
        != content_hash(
            residual,
            prefix="partition-free-v1-stratified-fd-residual-diagonal-",
        )
        or not same_geometric_coordinate(
            numerical_receipt.combined_residual_spectral_bound,
            geometry.residual_spectral_bound,
        )
    ):
        raise ValueError("sampled selector numerical assembly arrays do not replay")
    corrected_point_receipt_hash = content_hash(
        {
            "status": SAMPLED_CORRECTED_POINT_STATUS,
            "sampled_geometry_source_hash": source.source_hash,
            "corrected_moments_state_hash": admission_receipt.corrected_moments_state_hash,
            "sample_weight_hash": admission_receipt.sample_weight_hash,
            "corrected_path_cost_hash": path_hash,
            "signed_covector_hash": signed_hash,
            "finite_population_census": moments.is_finite_population_census,
            "population_exactness_authority": False,
        },
        prefix=_CORRECTED_POINT_PREFIX,
    )
    certificate = V1SampledSnapshotSelectorCertificate(
        _issuance_token=_ISSUANCE_TOKEN,
        sampled_geometry_source_hash=source.source_hash,
        admission_receipt_hash=admission_receipt.receipt_hash,
        assembly_receipt_hash=assembly_receipt.receipt_hash,
        campaign_contract_hash=admission_receipt.campaign_contract_hash,
        campaign_binding_hash=admission_receipt.campaign_binding_hash,
        model_identity_hash=admission_receipt.model_identity_hash,
        query_id=admission.binding.contract.query_id,
        query_contract_hash=admission_receipt.query_contract_hash,
        preprocessing_hash=admission_receipt.preprocessing_hash,
        sampling_law_hash=admission_receipt.sampling_law_hash,
        current_mask_hash=admission_receipt.current_mask_hash,
        proposal_generation=admission_receipt.proposal_generation,
        physical_route_axis_hash=admission_receipt.physical_route_axis_hash,
        ordered_route_axis_hash=admission_receipt.ordered_route_axis_hash,
        route_count=admission_receipt.route_count,
        sampled_look_index=admission_receipt.sampled_look_index,
        calibration_identity=admission_receipt.calibration_identity,
        raw_logit_stream_source_hash=(
            admission_receipt.raw_logit_stream_source_hash
        ),
        raw_logit_coordinate=admission_receipt.raw_logit_coordinate,
        context_mixture_hash=admission_receipt.context_mixture_hash,
        sample_weight_hash=admission_receipt.sample_weight_hash,
        source_measurement_axis_hash=(
            admission_receipt.source_measurement_axis_hash
        ),
        corrected_moments_state_hash=(
            admission_receipt.corrected_moments_state_hash
        ),
        corrected_path_cost_hash=path_hash,
        signed_covector_hash=signed_hash,
        corrected_point_receipt_hash=corrected_point_receipt_hash,
        fd_design_hash=admission_receipt.fd_design_hash,
        fd_snapshot_hash=admission_receipt.fd_snapshot_hash,
        snapshot_numerical_backend_hash=(
            admission_receipt.snapshot_numerical_backend_hash
        ),
        geometry_state_hash=geometry.content_hash,
        geometry_result_hash=geometry_result.result_hash,
        numerical_assembly_receipt_hash=numerical_receipt.receipt_hash,
        mean_component_hash=numerical_receipt.mean_component_hash,
        centered_factor_hash=numerical_receipt.centered_factor_hash,
        residual_diagonal_hash=residual_hash,
        low_rank_factor_hash=factor_hash,
        mean_row_count=int(mean.shape[0]),
        centered_rank=int(centered.shape[0]),
        residual_spectral_bound=geometry.residual_spectral_bound,
        raw_row_derivation_absence_hash=(
            assembly_receipt.raw_row_derivation_absence_hash
        ),
    )
    return V1SampledSnapshotSelectorNumerics(
        _issuance_token=_NUMERICS_ISSUANCE_TOKEN,
        certificate=certificate,
        corrected_path_cost=path,
        signed_covector=signed,
        low_rank_factor=factor,
        residual_diagonal=residual,
    )


@dataclass(frozen=True, slots=True, init=False)
class V1SampledProposalPositionCertificate:
    """Neutral ordinal-position correspondence for proposal construction.

    The sampled pre-freeze admission deliberately cannot apply an ordered
    action to physical mask bits.  This separate certificate proves only that
    the typed physical route declaration uses the same route-ID sequence, so
    selector coordinate ``i`` denotes physical mask bit ``i``.  It grants no
    irreversible application or downstream decision authority.
    """

    sampled_snapshot_certificate_hash: str
    sampled_geometry_source_hash: str
    current_mask_hash: str
    physical_route_axis_declaration_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    position_rule: str = SAMPLED_PROPOSAL_POSITION_RULE
    full_position_bijection_verified: bool = True
    current_mask_coordinate_bound: bool = True
    proposal_positional_correspondence_authority: bool = True
    pre_freeze_ordered_to_physical_application_authority: bool = False
    irreversible_order_authority: bool = False
    irreversible_application_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_execution_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = SAMPLED_PROPOSAL_POSITION_AUTHORITY_SCOPE
    schema: str = SAMPLED_PROPOSAL_POSITION_CERTIFICATE_SCHEMA

    def __init__(
        self,
        *,
        _issuance_token: object | None = None,
        **values: object,
    ) -> None:
        if _issuance_token is not _POSITION_ISSUANCE_TOKEN:
            raise TypeError(
                "sampled proposal-position certificates can only be issued from "
                "factory-prepared numerics and a DeclaredRouteAxis"
            )
        for item in self.__dataclass_fields__.values():
            if not item.init:
                continue
            value = values.pop(item.name, item.default)
            object.__setattr__(self, item.name, value)
        if values:
            raise TypeError(
                "unexpected proposal-position certificate fields: "
                f"{tuple(sorted(values))}"
            )
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "sampled_snapshot_certificate_hash",
            "sampled_geometry_source_hash",
            "current_mask_hash",
            "physical_route_axis_declaration_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "route_id_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="proposal-position route count"),
        )
        if self.physical_route_axis_hash == self.ordered_route_axis_hash:
            raise ValueError("proposal-position certificate must preserve distinct axes")
        if self.route_id_axis_hash != self.ordered_route_axis_hash:
            raise ValueError("proposal-position route-ID axis differs from ordered geometry")
        required_true = (
            self.full_position_bijection_verified,
            self.current_mask_coordinate_bound,
            self.proposal_positional_correspondence_authority,
        )
        required_false = (
            self.pre_freeze_ordered_to_physical_application_authority,
            self.irreversible_order_authority,
            self.irreversible_application_authority,
            self.fq_certification_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.production_execution_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in (*required_true, *required_false)):
            raise TypeError("proposal-position authority fields must be exact booleans")
        if not all(required_true) or any(required_false):
            raise PermissionError(
                "sampled proposal-position certificate crossed its neutral boundary"
            )
        if (
            self.position_rule != SAMPLED_PROPOSAL_POSITION_RULE
            or self.authority_scope != SAMPLED_PROPOSAL_POSITION_AUTHORITY_SCOPE
            or self.schema != SAMPLED_PROPOSAL_POSITION_CERTIFICATE_SCHEMA
        ):
            raise ValueError("unsupported sampled proposal-position certificate scope")

    @property
    def certificate_hash(self) -> str:
        return content_hash(self, prefix=_POSITION_CERTIFICATE_PREFIX)


def certify_v1_sampled_proposal_position_correspondence(
    *,
    numerics: V1SampledSnapshotSelectorNumerics,
    physical_route_axis: DeclaredRouteAxis,
) -> V1SampledProposalPositionCertificate:
    """Certify a neutral positional bijection without reusing score authority."""

    if type(numerics) is not V1SampledSnapshotSelectorNumerics:
        raise TypeError("proposal-position certification requires exact sampled numerics")
    if type(physical_route_axis) is not DeclaredRouteAxis:
        raise TypeError("proposal-position certification requires a DeclaredRouteAxis")
    certificate = numerics.certificate
    if (
        physical_route_axis.route_count != certificate.route_count
        or physical_route_axis.route_axis_hash
        != certificate.physical_route_axis_hash
        or physical_route_axis.route_id_axis_hash
        != certificate.ordered_route_axis_hash
    ):
        raise ValueError(
            "declared physical route positions differ from sampled selector coordinates"
        )
    return V1SampledProposalPositionCertificate(
        _issuance_token=_POSITION_ISSUANCE_TOKEN,
        sampled_snapshot_certificate_hash=certificate.certificate_hash,
        sampled_geometry_source_hash=certificate.sampled_geometry_source_hash,
        current_mask_hash=certificate.current_mask_hash,
        physical_route_axis_declaration_hash=physical_route_axis.content_hash,
        physical_route_axis_hash=physical_route_axis.route_axis_hash,
        ordered_route_axis_hash=physical_route_axis.route_id_axis_hash,
        route_id_axis_hash=physical_route_axis.route_id_axis_hash,
        route_count=physical_route_axis.route_count,
    )


@dataclass(frozen=True, slots=True)
class V1SampledSelectorChartCoordinates:
    """Common current-chart coordinates exposed to freeze and selector."""

    sampled_snapshot_certificate_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    campaign_binding_hash: str
    model_seal_hash: str
    query_id: str
    query_contract_hash: str
    preprocessing_hash: str
    sampling_law_hash: str
    train_bank_source_hash: str
    calibration_identity: str
    proposal_generation: int
    schema: str = SAMPLED_SELECTOR_CHART_COORDINATES_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "sampled_snapshot_certificate_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "campaign_binding_hash",
            "model_seal_hash",
            "query_id",
            "query_contract_hash",
            "preprocessing_hash",
            "sampling_law_hash",
            "train_bank_source_hash",
            "calibration_identity",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        if self.physical_route_axis_hash == self.ordered_route_axis_hash:
            raise ValueError("sampled selector chart axes must remain distinct")
        if self.schema != SAMPLED_SELECTOR_CHART_COORDINATES_SCHEMA:
            raise ValueError("unsupported sampled selector chart coordinates schema")

    @property
    def coordinates_hash(self) -> str:
        return content_hash(self, prefix=_COORDINATES_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SampledScoreDiscoveryCoordinates:
    """Opaque, proposal-only identities from the registered D look."""

    registration_hash: str
    design_hash: str
    look_receipt_hash: str
    result_hash: str
    statistics_hash: str
    evidence_hash: str
    look_index: int
    proposal_only: bool = True
    score_zero_regret_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = SAMPLED_SCORE_DISCOVERY_COORDINATES_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "design_hash",
            "look_receipt_hash",
            "result_hash",
            "statistics_hash",
            "evidence_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "look_index",
            _positive_int(self.look_index, name="score discovery look index"),
        )
        for name in (
            "proposal_only",
            "score_zero_regret_authority",
            "mask_commit_authority",
            "terminal_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if (
            self.proposal_only is not True
            or self.score_zero_regret_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            raise PermissionError("sampled score discovery crossed its authority scope")
        if self.schema != SAMPLED_SCORE_DISCOVERY_COORDINATES_SCHEMA:
            raise ValueError("unsupported sampled score discovery coordinates schema")

    @property
    def coordinates_hash(self) -> str:
        return content_hash(self, prefix=_DISCOVERY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SampledSelectorChartProvenance:
    """Typed sampled current-chart owner for canonical proposal selection."""

    snapshot_certificate: V1SampledSnapshotSelectorCertificate = field(
        repr=False,
        compare=False,
    )
    proposal_position_certificate: V1SampledProposalPositionCertificate = field(
        repr=False,
        compare=False,
    )
    corrected_attribution_coordinates: V1SampledSelectorChartCoordinates
    score_discovery_coordinates: V1SampledScoreDiscoveryCoordinates
    geometric_trust_budgets: V1GeometricTrustBudgets
    fixed_affine_bce_calibration: FixedAffineBCECalibration
    resource_plane: str
    resource_cost_policy_hash: str
    resource_cost_hash: str
    eligibility_receipt_hash: str
    legality_policy_hash: str
    proposal_only: bool = True
    sampled_geometry_statistical_optimality_claim: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = SAMPLED_SELECTOR_CHART_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        certificate = self.snapshot_certificate
        position = self.proposal_position_certificate
        coordinates = self.corrected_attribution_coordinates
        discovery = self.score_discovery_coordinates
        budgets = self.geometric_trust_budgets
        calibration = self.fixed_affine_bce_calibration
        if type(certificate) is not V1SampledSnapshotSelectorCertificate:
            raise TypeError("sampled chart requires its exact snapshot certificate")
        if type(position) is not V1SampledProposalPositionCertificate:
            raise TypeError("sampled chart requires its exact proposal-position certificate")
        if type(coordinates) is not V1SampledSelectorChartCoordinates:
            raise TypeError("sampled chart requires its exact coordinates")
        if type(discovery) is not V1SampledScoreDiscoveryCoordinates:
            raise TypeError("sampled chart requires exact score-discovery coordinates")
        if type(budgets) is not V1GeometricTrustBudgets:
            raise TypeError("sampled chart requires typed geometric trust budgets")
        if type(calibration) is not FixedAffineBCECalibration:
            raise TypeError("sampled chart requires one fixed affine BCE calibration")
        shared = (
            (coordinates.sampled_snapshot_certificate_hash, certificate.certificate_hash),
            (position.sampled_snapshot_certificate_hash, certificate.certificate_hash),
            (position.sampled_geometry_source_hash, certificate.sampled_geometry_source_hash),
            (position.current_mask_hash, certificate.current_mask_hash),
            (position.physical_route_axis_hash, certificate.physical_route_axis_hash),
            (position.ordered_route_axis_hash, certificate.ordered_route_axis_hash),
            (position.route_count, certificate.route_count),
            (coordinates.current_mask_hash, certificate.current_mask_hash),
            (coordinates.physical_route_axis_hash, certificate.physical_route_axis_hash),
            (coordinates.ordered_route_axis_hash, certificate.ordered_route_axis_hash),
            (coordinates.campaign_binding_hash, certificate.campaign_binding_hash),
            (coordinates.model_seal_hash, certificate.model_identity_hash),
            (coordinates.query_id, certificate.query_id),
            (coordinates.query_contract_hash, certificate.query_contract_hash),
            (coordinates.preprocessing_hash, certificate.preprocessing_hash),
            (coordinates.calibration_identity, certificate.calibration_identity),
            (coordinates.proposal_generation, certificate.proposal_generation),
            (discovery.statistics_hash, certificate.corrected_moments_state_hash),
            (discovery.look_index, certificate.sampled_look_index),
            (budgets.current_mask_hash, certificate.current_mask_hash),
            (calibration.calibration_state_id, certificate.calibration_identity),
            (calibration.query_id, certificate.query_id),
        )
        if any(left != right for left, right in shared):
            raise ValueError("sampled selector chart coordinates do not close")
        if not same_geometric_coordinate(
            budgets.raw_logit_to_calibrated_logit_slope,
            calibration.scale,
        ):
            raise ValueError("sampled selector budget/calibration slopes differ")
        if not same_geometric_coordinate(
            budgets.quadratic_trust_energy_budget,
            budgets.metric_length_budget * budgets.metric_length_budget,
        ):
            raise ValueError("sampled selector tau_Q square law differs")
        for name in (
            "resource_plane",
            "resource_cost_policy_hash",
            "resource_cost_hash",
            "eligibility_receipt_hash",
            "legality_policy_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "proposal_only",
            "sampled_geometry_statistical_optimality_claim",
            "mask_commit_authority",
            "terminal_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if (
            self.proposal_only is not True
            or self.sampled_geometry_statistical_optimality_claim is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            raise PermissionError("sampled selector chart crossed its proposal boundary")
        if self.schema != SAMPLED_SELECTOR_CHART_PROVENANCE_SCHEMA:
            raise ValueError("unsupported sampled selector chart provenance schema")

    @property
    def provenance_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "snapshot_certificate_hash": self.snapshot_certificate.certificate_hash,
                "proposal_position_certificate_hash": (
                    self.proposal_position_certificate.certificate_hash
                ),
                "corrected_attribution_coordinates_hash": (
                    self.corrected_attribution_coordinates.coordinates_hash
                ),
                "score_discovery_coordinates_hash": (
                    self.score_discovery_coordinates.coordinates_hash
                ),
                "geometric_trust_budget_hash": self.geometric_trust_budgets.budget_hash,
                "fixed_affine_bce_calibration_hash": (
                    self.fixed_affine_bce_calibration.content_hash
                ),
                "resource_plane": self.resource_plane,
                "resource_cost_policy_hash": self.resource_cost_policy_hash,
                "resource_cost_hash": self.resource_cost_hash,
                "eligibility_receipt_hash": self.eligibility_receipt_hash,
                "legality_policy_hash": self.legality_policy_hash,
                "proposal_only": True,
                "sampled_geometry_statistical_optimality_claim": False,
                "mask_commit_authority": False,
                "terminal_authority": False,
            },
            prefix=_PROVENANCE_PREFIX,
        )


def build_v1_sampled_selector_chart_provenance(
    *,
    numerics: V1SampledSnapshotSelectorNumerics,
    proposal_position_certificate: V1SampledProposalPositionCertificate,
    score_discovery_coordinates: V1SampledScoreDiscoveryCoordinates,
    geometric_trust_budgets: V1GeometricTrustBudgets,
    fixed_affine_bce_calibration: FixedAffineBCECalibration,
    train_bank_source_hash: str,
    resource_plane: str,
    resource_cost_policy_hash: str,
    eligibility_receipt_hash: str,
    legality_policy_hash: str,
    uniform_resource_cost: float | None,
    resource_cost: np.ndarray | None,
) -> V1SampledSelectorChartProvenance:
    """Build one sampled chart without granting downstream authority."""

    if type(numerics) is not V1SampledSnapshotSelectorNumerics:
        raise TypeError("sampled chart builder requires exact prepared numerics")
    if type(proposal_position_certificate) is not V1SampledProposalPositionCertificate:
        raise TypeError("sampled chart builder requires exact positional correspondence")
    certificate = numerics.certificate
    coordinates = V1SampledSelectorChartCoordinates(
        sampled_snapshot_certificate_hash=certificate.certificate_hash,
        current_mask_hash=certificate.current_mask_hash,
        physical_route_axis_hash=certificate.physical_route_axis_hash,
        ordered_route_axis_hash=certificate.ordered_route_axis_hash,
        campaign_binding_hash=certificate.campaign_binding_hash,
        model_seal_hash=certificate.model_identity_hash,
        query_id=certificate.query_id,
        query_contract_hash=certificate.query_contract_hash,
        preprocessing_hash=certificate.preprocessing_hash,
        sampling_law_hash=certificate.sampling_law_hash,
        train_bank_source_hash=_identity(
            train_bank_source_hash,
            name="train bank source hash",
        ),
        calibration_identity=certificate.calibration_identity,
        proposal_generation=certificate.proposal_generation,
    )
    return V1SampledSelectorChartProvenance(
        snapshot_certificate=certificate,
        proposal_position_certificate=proposal_position_certificate,
        corrected_attribution_coordinates=coordinates,
        score_discovery_coordinates=score_discovery_coordinates,
        geometric_trust_budgets=geometric_trust_budgets,
        fixed_affine_bce_calibration=fixed_affine_bce_calibration,
        resource_plane=resource_plane,
        resource_cost_policy_hash=resource_cost_policy_hash,
        resource_cost_hash=_resource_cost_hash(
            route_count=certificate.route_count,
            uniform_resource_cost=uniform_resource_cost,
            resource_cost=resource_cost,
        ),
        eligibility_receipt_hash=eligibility_receipt_hash,
        legality_policy_hash=legality_policy_hash,
    )


__all__ = [
    "SAMPLED_CORRECTED_PATH_HASH_PREFIX",
    "SAMPLED_PROPOSAL_POSITION_AUTHORITY_SCOPE",
    "SAMPLED_PROPOSAL_POSITION_CERTIFICATE_SCHEMA",
    "SAMPLED_PROPOSAL_POSITION_RULE",
    "SAMPLED_SCORE_DISCOVERY_COORDINATES_SCHEMA",
    "SAMPLED_SELECTOR_CHART_COORDINATES_SCHEMA",
    "SAMPLED_SELECTOR_CHART_PROVENANCE_SCHEMA",
    "SAMPLED_SIGNED_COVECTOR_HASH_PREFIX",
    "SAMPLED_SNAPSHOT_PROVENANCE_SCOPE",
    "SAMPLED_SNAPSHOT_SELECTOR_CERTIFICATE_SCHEMA",
    "SAMPLED_SNAPSHOT_SELECTOR_NUMERICS_SCHEMA",
    "V1SampledScoreDiscoveryCoordinates",
    "V1SampledProposalPositionCertificate",
    "V1SampledSelectorChartCoordinates",
    "V1SampledSelectorChartProvenance",
    "V1SampledSnapshotSelectorCertificate",
    "V1SampledSnapshotSelectorNumerics",
    "build_v1_sampled_selector_chart_provenance",
    "certify_v1_sampled_proposal_position_correspondence",
    "prepare_v1_sampled_snapshot_selector_numerics",
]
