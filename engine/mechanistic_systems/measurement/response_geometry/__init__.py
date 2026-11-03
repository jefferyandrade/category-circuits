"""Lazy facade for workflow-neutral response measurement representations."""

from __future__ import annotations

from importlib import import_module


_EXPORT_GROUPS = {
    ".contracts": (
        "readonly_bool_vector",
        "readonly_matrix",
        "readonly_vector",
        "validate_route_axis",
    ),
    ".coupled": ("CoupledResponseField",),
    ".category_contrast": (
        "CategoryContrastJacobian",
        "CategoryContrastResponse",
        "CategoryContrastResponseSpec",
        "ExactMaskResponseComparison",
        "ExactMaskResponseReceipt",
        "build_exact_mask_response_receipt",
        "category_contrast_jacobian",
        "category_contrast_response",
        "compare_exact_mask_logits",
    ),
    ".intrinsic": ("IntrinsicResponseField",),
    ".quotient": (
        "IntrinsicResponseFactorOperator",
        "ResponseQuotientDiagnostics",
    ),
    ".transitions": (
        "FiniteTransitionResponseField",
        "legacy_v4_state_hash",
        "legacy_v4_state_payload",
        "positive_semidefinite_part",
        "psd_common_upper_envelope",
        "readonly_optional_matrix",
        "readonly_optional_psd",
        "readonly_optional_tensor3",
        "uniform_sample_mean_variance_from_mean_rms",
    ),
    ".tensors": (
        "ActionResponseMoments",
        "RESPONSE_TRANSITION_CACHE_KEY",
        "ResponseActionMoments",
        "ResponseConstraintEvaluator",
        "ResponseEllipsoidTrustRegion",
        "ResponseEllipsoidTrustRegionDecision",
        "ResponseFactorTensor",
        "ResponseModelValidityReceipt",
        "ResponseMomentFactor",
        "ResponseMomentSummary",
        "ResponseMomentTensor",
        "ResponseProjectionReceipt",
        "ResponseRiskEvaluation",
        "ResponseTransitionReceipt",
        "TargetCompetitorMomentDecomposition",
        "response_transition_receipt_from_outcome",
    ),
}

_LAZY_EXPORTS = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
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
