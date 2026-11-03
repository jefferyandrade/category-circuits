"""Lazy facade for workflow-neutral experimental-design capabilities."""

from __future__ import annotations

from importlib import import_module


_EXPORT_GROUPS = {
    ".action_library": ("PairwiseActionLibraryReceipt",),
    ".finite_response_contexts": (
        "FINITE_RESPONSE_CONTEXT_AUTHORITY_CEILING", "FiniteResponsePacketPolicy",
        "ResponseStratumPopulation", "compile_finite_response_context_design",
    ),
    ".remaining_information": (
        "REMAINING_INFORMATION_AUTHORITY_CEILING", "InformationResolutionThresholds",
        "RemainingInformationDiagnostic", "localized_response_discrepancy_radius",
        "normalized_epistemic_uncertainty",
    ),
    ".sampling_identity": (
        "BINARY_EVIDENCE_ROLE_STRATA", "DISJOINT_ROLE_DESIGN_AUTHORITY_CEILING",
        "BinaryEvidenceRoleDesign", "BinaryIIDSamplingDesign", "IIDPhysicalPacketPolicy",
        "binary_evidence_role_namespace", "compile_binary_iid_sampling_design",
        "compile_disjoint_binary_evidence_role_design",
        "naturalize_iid_physical_packet_policy", "sealed_evidence_role_order",
    ),
    ".screening": ("ResponseScreenReceipt",),
    ".response_acquisition": (
        "CALIBRATION_FORECAST_AUTHORITY_CEILING", "RESPONSE_ACQUISITION_AUTHORITY_CEILING",
        "AcquisitionScore", "CalibrationInformationForecast", "CalibrationSolveReceipt",
        "CandidateProposalReceipt", "CandidateSearchSurface", "MeasurementDesignForecast",
        "MeasurementEvidenceRole", "MeasurementModality", "MeasurementOpportunity",
        "MeasurementOpportunitySurface", "MeasurementProposalReceipt",
        "MeasurementRankingReceipt", "propose_candidate_batch_with_scores",
        "propose_from_candidate_surface", "propose_from_measurement_surface",
        "rank_measurement_opportunities", "score_response_candidate",
    ),
    ".response_candidates": (
        "LEGACY_V5_CANDIDATE_COMPATIBILITY_CEILING",
        "RESPONSE_CANDIDATE_AUTHORITY_CEILING", "CandidateExperiment", "CandidatePool",
        "CandidatePoolReceipt", "build_candidate_pool", "candidate_fingerprint",
        "legacy_v5_candidate_fingerprint",
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
