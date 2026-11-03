"""Lazy facade for canonical scientific and numerical contracts."""

from __future__ import annotations

from importlib import import_module


_EXPORT_GROUPS = {
    ".content_address": ("canonical_json", "content_hash"),
    ".covariance": (
        "CovarianceCorrectionPolicy", "CovarianceCorrectionReceipt",
        "canonicalize_covariance", "canonicalize_covariance_with_receipt",
    ),
    ".calibration": (
        "CALIBRATION_LOSS_FAMILIES", "CALIBRATION_PARAMETERIZATIONS",
        "QUERY_OPERATOR_KINDS", "READOUT_PROJECTION_KINDS",
        "CalibrationProblemSpec", "QueryOperatorSpec", "ReadoutProjectionSpec",
    ),
    ".measurement": (
        "ArtifactRef", "ContextSelector", "MeasurementKind", "MeasurementOutcome",
        "MeasurementProvenance", "MeasurementProvider", "MeasurementRequest",
        "OutcomeCache", "replace_request_value",
    ),
    ".measurement_costs": ("CostEstimate", "RealizedCost"),
    ".interventions": ("HardMaskView",),
    ".functional_atlas": ("FunctionalAtlasSpec",),
    ".response_geometry": (
        "HistoricalResponseProfileName", "ResponseFunctionalSpec",
        "ResponseMetricSpec",
    ),
    ".trust_geometry": ("GlobalTrustGeometrySpec", "ModelValidityDecision"),
    ".three_program": (
        "ArrayArtifactBinding", "ControlCircuitIdentity", "ControlCircuitRole",
        "EvidenceCeiling", "EvidenceCeilingReceipt",
        "ExecutedHierarchyResponseArtifact", "FactorialPopulationReceipt",
        "FrozenCircuitPopulationManifest", "ThreeProgramUnitIdentity",
        "certify_complete_factorial_population", "require_same_stimulus_axis",
        "require_same_unit_identity",
    ),
    ".operator_route_v2": (
        "ActivationConeSpec", "AdaptivePacketSpec", "CalibrationProfileSpec",
        "FOUR_CONTEXT_BLOCKS", "MetricOperatorSpec", "OutputTangentSpec",
        "PruningObjectiveSpec", "RouteGeometrySpec", "RouteRoleSpec",
        "RouteTopologySpec", "SCALAR_REFERENCE_SCHEMA", "SelectionProtocolSpec",
        "SelectionUnitAssignment", "TopologyNodeSpec",
    ),
    ".operator_route_views": ("ReferenceSearchResultView", "WinnerLockView"),
    ".science": (
        "AnalysisPlan", "CANONICAL_CIRCUIT_ENGINE",
        "CANONICAL_CIRCUIT_SEARCH_ALGORITHMS",
        "CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON",
        "CANONICAL_CALIBRATION_HESSIAN_RIDGE",
        "CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS",
        "CANONICAL_CALIBRATION_KKT_TOLERANCE",
        "CANONICAL_CALIBRATION_MAX_ITERATIONS",
        "CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE",
        "CANONICAL_CALIBRATION_REFERENCE_TOLERANCE",
        "CANONICAL_CALIBRATION_SELECTION_BLOCKS", "CANONICAL_PARTITION_POLICIES",
        "CANONICAL_PRUNING_CALIBRATION_ALGORITHM", "CANONICAL_TARGET_CALIBRATION_MODE",
        "CircuitPlan", "DataView", "GeometryPlan", "InterventionPlan",
        "KernelContractionSpec", "MeasurementPlan", "MechanismPlan",
        "PRIMARY_MEASUREMENT_STATISTICS", "PRODUCTION_CIRCUIT_SEARCH_ALGORITHM",
        "RATE_DISTORTION_V2", "PROPER_SCORE_SHRINKAGE_AFFINE_V2",
        "RISK_AWARE_ROUTE_GATE_V2", "ROUTE_GATE_V1", "SAMPLED_PATH_FRONTIER_V1",
        "SAMPLED_PATH_GROUP_REFERENCE_POLICY", "SAMPLED_PATH_V1_CALIBRATION_ALGORITHM",
        "QuerySpec", "RouteCoordinateSchema", "RuntimePlan", "StudySpec",
        "canonical_calibration_policy", "ordered_axis_hash",
    ),
}

_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = list(_LAZY_EXPORTS)
