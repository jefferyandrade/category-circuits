"""Reporting-only statistics for train-frozen terminal calibration.

All risks use natural logarithms and the frozen pooled affine recorded in the
outer receipt (also when older optional crossfit ownership fields exist).
Each validation image has equal mass in
``per_image_bce``; ``class_balanced_bce`` gives the target half the mass and
each represented off-target class an equal share of the other half, with
equal row weights within classes. ``empty_circuit_reference`` reports both
risks for the constant prediction at the calibration prior: training pi for
per_image and 1/2 for balanced. No validation prevalence is used in a fit.
For the substituted functional the reference instead fixes target scale to
zero and fits its intercept on the same training rows, penalty and weighting:
``eta_i = b - logsumexp(dense_competitor_logits_i)``. Its two risks are
evaluated on the validation rows; its fitted intercept and convergence are
recorded. The substituted fit's attainable-improvement bounds use its own
training rows and fitted parameters and are diagnostics only.
When supplied, ``sample_weights`` recover population image means from a
capped validation panel for per-image risks and error rates. The conditional
TPR/FPR and explicitly class-balanced diagnostics retain their definitions.

``likelihood_ratio_1`` calls target at calibrated logit >= logit(prior).
``probability_0_5`` calls target at calibrated logit >= 0. Both report TPR
(target-row mean), FPR (mean of off-target class means), and Youden J=TPR-FPR;
the latter also reports errors divided by the number of validation images.
``network_top_1`` uses the train-frozen substituted target score against the
dense competitor scores; ties go to the lowest class index, as in argmax.
``network_probability_0_5`` uses the substituted target score >= competitor
log-sum-exp. These network decisions report hits_retained=TPR,
false_alarms=FPR, and J_net=TPR-FPR. Network fields occur in the substituted
functional's block; ordinary evaluation has no dense competitor readout.

AUC uses the uncalibrated target logit, averaging target-versus-each-class
pair probabilities with half credit for ties. Dense and final masks each
carry their own block. Thus (final risk - dense risk)/(empty risk - dense
risk) is computable for either risk without another model evaluation.
If the denominator vanishes, that loss-window share is undefined.

The implementation extends the native terminal evaluator; it neither fits
calibration nor participates in walk decisions. Tests use direct pairwise
comparisons and NumPy argmax as references rather than these reducers.
"""

from typing import Any, Mapping, Optional

import numpy as np


def terminal_statistics_enabled(weighting: str, penalty: str) -> bool:
    """Keep the legacy balanced/identity-anchor receipt surface unchanged."""
    return weighting == "per_image" or penalty == "firth"


def terminal_statistics_v1(
    *, raw_target_logits: Any, calibrated_logits: Any, class_ids: Any,
    target_class_id: int, training_target_prevalence: float,
    loss_weighting: str, validation_counts_by_stratum: Mapping[str, int],
    auc: Mapping[str, Any], network_target_selected: Optional[Any] = None,
    empty_circuit_logits: Optional[Any] = None,
    empty_circuit_fit: Optional[Mapping[str, Any]] = None,
    calibration_improvement: Optional[Mapping[str, Any]] = None,
    sample_weights: Optional[Any] = None,
) -> dict:
    """Reduce existing validation logits; no model or fitting dependencies."""
    raw = np.asarray(raw_target_logits, dtype=np.float64).reshape(-1)
    eta = np.asarray(calibrated_logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    pi = float(training_target_prevalence)
    if (raw.shape != eta.shape or raw.shape != labels.shape or not raw.size
            or not np.isfinite(raw).all() or not np.isfinite(eta).all()
            or not 0.0 < pi < 1.0 or loss_weighting not in ("per_image", "balanced")):
        raise ValueError("terminal statistics require aligned finite logits and a training prevalence")
    counts = {str(key): int(value) for key, value in validation_counts_by_stratum.items()}
    if any(value <= 0 for value in counts.values()) or sum(counts.values()) != raw.size:
        raise ValueError("terminal validation counts must cover every row")
    on = labels == int(target_class_id)
    off_classes = np.unique(labels[~on])
    if not on.any() or not off_classes.size:
        raise ValueError("terminal statistics require target and off-target classes")
    prior = pi if loss_weighting == "per_image" else 0.5
    threshold = float(np.log(prior) - np.log1p(-prior))
    image_weights = None
    if sample_weights is not None:
        image_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if (image_weights.shape != raw.shape or not np.isfinite(image_weights).all()
                or np.any(image_weights < 0.0) or not image_weights.sum() > 0.0):
            raise ValueError("terminal per-image weights must be aligned, finite and nonnegative with positive mass")
        if not np.isfinite(image_weights.sum()):
            raise ValueError("terminal per-image weights require finite total mass")
        image_weights = image_weights / image_weights.sum()

    def image_mean(values):
        return float(np.mean(values) if image_weights is None else np.sum(image_weights * values))

    def balanced_mean(values):
        return float(0.5 * np.mean(values[on]) + 0.5 * np.mean([
            np.mean(values[labels == label]) for label in off_classes]))

    def decision(selected, *, network=False, error=False):
        selected = np.asarray(selected, dtype=bool).reshape(-1)
        if selected.shape != labels.shape:
            raise ValueError("terminal decisions must cover every validation row")
        tpr = float(np.mean(selected[on]))
        fpr = float(np.mean([np.mean(selected[labels == label]) for label in off_classes]))
        result = ({"hits_retained": tpr, "false_alarms": fpr, "J_net": tpr - fpr}
                  if network else {"tpr": tpr, "fpr": fpr, "youden_j": tpr - fpr})
        if error:
            result["per_image_error_rate"] = image_mean(selected != on)
        return result

    bce = np.logaddexp(0.0, np.where(on, -eta, eta))
    empty_bce = np.where(on, -np.log(prior), -np.log1p(-prior))
    result = {
        "schema_name": "terminal_statistics_v1",
        "loss_weighting": loss_weighting,
        "training_target_prevalence": pi,
        "calibration_prior": prior,
        "per_image_bce": image_mean(bce),
        "class_balanced_bce": balanced_mean(bce),
        "empty_circuit_reference": {
            "prediction": prior,
            "per_image_bce": image_mean(empty_bce),
            "class_balanced_bce": balanced_mean(empty_bce),
        },
        "auc": dict(auc),
        "likelihood_ratio_1": dict(decision(eta >= threshold), calibrated_logit_cut=threshold),
        "probability_0_5": dict(decision(eta >= 0.0, error=True), calibrated_logit_cut=0.0),
        "validation_image_count": int(raw.size),
        "validation_target_image_count": int(on.sum()),
        "validation_off_target_image_count": int((~on).sum()),
        "validation_counts_by_stratum": counts,
        "validation_counts_by_class": {str(int(label)): int(np.sum(labels == label))
                                       for label in np.unique(labels)},
        "off_target_class_count": int(off_classes.size),
    }
    if network_target_selected is not None:
        result["network_top_1"] = dict(decision(network_target_selected, network=True),
                                        tie_rule="lowest_class_index")
        result["network_probability_0_5"] = decision(eta >= 0.0, network=True)
    if empty_circuit_logits is not None:
        empty_eta = np.asarray(empty_circuit_logits, dtype=np.float64).reshape(-1)
        if empty_eta.shape != eta.shape or not np.isfinite(empty_eta).all() or empty_circuit_fit is None:
            raise ValueError("substituted empty reference requires finite aligned logits and its frozen fit")
        empty_loss = np.logaddexp(0.0, np.where(on, -empty_eta, empty_eta))
        result["empty_circuit_reference"] = dict(
            empty_circuit_fit, per_image_bce=image_mean(empty_loss),
            class_balanced_bce=balanced_mean(empty_loss))
    elif empty_circuit_fit is not None:
        if empty_circuit_fit.get("converged") is not False:
            raise ValueError("substituted empty reference fit requires its validation logits")
        result["empty_circuit_reference"] = dict(
            empty_circuit_fit, per_image_bce=None, class_balanced_bce=None)
    if calibration_improvement is not None:
        result.update(calibration_improvement)
    return result


def substituted_loss_window_statistics(metrics: Mapping[str, Any], dense_metrics: Mapping[str, Any]) -> dict:
    """Attach reporting shares after both substituted evaluations exist.

    Each share is (circuit BCE - dense BCE)/(empty-reference BCE - dense BCE),
    using the corresponding validation risk. A zero denominator gives None.
    Older receipt surfaces pass through unchanged.
    """
    result = dict(metrics)
    stats = metrics.get("terminal_statistics_v1")
    dense = dense_metrics.get("terminal_statistics_v1")
    if not isinstance(stats, Mapping):
        return result
    if not isinstance(dense, Mapping):
        if metrics.get("loss_weighting") == "per_image":
            result["terminal_statistics_v1"] = dict(stats, loss_window_share={
                "definition": "(circuit_bce - dense_bce) / (empty_circuit_reference_bce - dense_bce)",
                "per_image_bce": None, "class_balanced_bce": None})
        return result
    reference = stats.get("empty_circuit_reference", {})
    if not isinstance(reference, Mapping) or reference.get("scale") != 0.0:
        return result
    shares = {"definition": "(circuit_bce - dense_bce) / (empty_circuit_reference_bce - dense_bce)"}
    for key in ("per_image_bce", "class_balanced_bce"):
        if any(block.get(key) is None for block in (reference, dense, stats)):
            shares[key] = None
        else:
            denominator = float(reference[key]) - float(dense[key])
            shares[key] = None if denominator == 0.0 else (float(stats[key]) - float(dense[key])) / denominator
    result["terminal_statistics_v1"] = dict(stats, loss_window_share=shares)
    return result
