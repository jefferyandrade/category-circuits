"""Lazy facade for cross-fitted and response-landscape inference."""

from __future__ import annotations

from importlib import import_module


_EXPORT_GROUPS = {
    ".conditional_gain": (
        "ConditionalGainFold", "ConditionalGainResult", "anchor_bootstrap_interval",
        "cross_fitted_relational_gain", "hierarchical_fdr",
    ),
    ".artifact": ("RELATIONAL_INFERENCE_SCHEMA", "materialize_relational_inference_artifact"),
    ".cross_fitting": ("AxisFoldAssignment", "make_block_folds"),
    ".equivalence": (
        "EquivalenceMetric", "bootstrap_kernel_equivalence", "centered_kernel_alignment",
        "equivalence_audit", "kernel_projector_distance",
    ),
    ".kernel": (
        "NestedFoldAssignments", "compile_nested_fold_assignments",
        "select_residualized_kernel_hyperparameters",
    ),
    ".operator_route_v2": (
        "PreselectionLineageReceipt", "SelectionInputRef", "SelectionInputRole",
        "certify_preselection_lineage",
    ),
    ".response_landscape": (
        "CallableKernelAdapter", "CompositeFeatureMap",
        "ContextCovarianceFieldPosterior", "CovarianceCoefficientEncoding",
        "MechanismFeatureMap", "MultiOutputFeaturePosterior", "NystromFeatureBlock",
        "PosteriorCriticismReport", "RBFKernel", "evaluate_posterior_criticism",
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
