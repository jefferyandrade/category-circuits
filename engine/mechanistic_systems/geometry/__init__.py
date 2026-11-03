"""Geometry capabilities with a cycle-free lazy public facade.

Importing a numerical metric or projector must not initialize storage,
CircuitFlow, Torch execution, or circuit-search packages.  Public behavior is
preserved through lazy attribute resolution while dependency direction remains
from those higher-level capabilities toward this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


PRIMARY_ANATOMY_METRICS = ("binary_cosine", "jaccard")
SENSITIVITY_ANATOMY_METRICS = ("weighted_cosine", "containment_mean")
DIAGNOSTIC_ANATOMY_METRICS = ("containment_directional",)

_PUBLIC_MODULES: Final[dict[str, str]] = {
    "CategoryContrastJacobian": ".category_contrast",
    "CategoryContrastResponse": ".category_contrast",
    "CategoryContrastResponseSpec": ".category_contrast",
    "ExactMaskResponseComparison": ".category_contrast",
    "ExactMaskResponseReceipt": ".category_contrast",
    "build_exact_mask_response_receipt": ".category_contrast",
    "category_contrast_jacobian": ".category_contrast",
    "category_contrast_response": ".category_contrast",
    "compare_exact_mask_logits": ".category_contrast",
    "contract_response_basis": ".operator",
    "double_center_kernel": ".operator",
    "materialize_mechanism_geometry": ".operator",
    "query_coupled_kernel": ".operator",
    "STRUCTURAL_GEOMETRY_BANK_SCHEMA": ".structural",
    "StructuralContractionResult": ".structural",
    "build_structural_geometry": ".structural",
    "contract_circuit_anatomy": ".structural",
    "materialize_structural_geometry": ".structural",
    "GeometryBankReader": ".storage",
    "LoadedGeometryOperator": ".storage",
    "load_geometry_operator": ".storage",
    "write_geometry_bank_zarr": ".storage",
    "validate_metric": ".metrics",
    "IdentificationPolicy": ".quotient",
    "IntrinsicMetricOperator": ".quotient",
    "PseudoinverseSpectrum": ".quotient",
    "QuotientDiagnostics": ".quotient",
    "pseudoinverse_spectrum": ".quotient",
    "ProjectorMatch": ".projectors",
    "StableProjector": ".projectors",
    "match_projectors": ".projectors",
    "principal_angles": ".projectors",
    "residualize_route_direction": ".projectors",
    "stable_spectral_projectors": ".projectors",
    "RESPONSE_CHART_AUTHORITY_CEILING": ".response_chart",
    "THRESHOLDED_MASK_PROPOSAL_AUTHORITY_CEILING": ".response_chart",
    "RelaxedMechanismChart": ".response_chart",
    "RelaxedResponseChart": ".response_chart",
    "ThresholdedMaskProposal": ".response_chart",
    "chart_from_response_factor": ".response_chart",
    "LEGACY_CHART_ADAPTER_AUTHORITY_CEILING": ".legacy_chart_adapter",
    "LegacyHardConcreteChartAdapter": ".legacy_chart_adapter",
    "frozen_chart_from_legacy_evaluation": ".legacy_chart_adapter",
    "FrequentDirectionsTrustAccumulator": ".trust",
    "GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID": ".trust",
    "GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID": ".trust",
    "GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA": ".trust",
    "GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA": ".trust",
    "GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY": ".trust",
    "GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY": ".trust",
    "GlobalTrustGeometryDerivationReceipt": ".trust",
    "GlobalTrustGeometryState": ".trust",
    "GlobalTrustRankGrowthObligation": ".trust",
    "global_trust_ordered_source_rows_hash": ".trust",
    "global_trust_source_row_hash": ".trust",
    "global_trust_stream_weight_hash": ".trust",
    "register_rank_growth_obligation": ".trust",
}


def __getattr__(name: str):
    """Resolve one public capability without importing unrelated owners."""

    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC_MODULES))


__all__ = [
    "DIAGNOSTIC_ANATOMY_METRICS",
    "PRIMARY_ANATOMY_METRICS",
    "SENSITIVITY_ANATOMY_METRICS",
    *_PUBLIC_MODULES,
]
