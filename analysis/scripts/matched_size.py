"""Helpers of the frontier comparison (frontier_analysis.py, Fig. 3A): a circuit's receipt read into the compared
measures, bootstrap intervals of the paired differences with Bonferroni family intervals, and the dominance rule.
The intervals use the category bootstrap of analysis/stats/category_resampling.py.
"""
import math
from pathlib import Path
import sys

import numpy as np

STATS_DIR = Path(__file__).resolve().parents[1] / 'stats'
sys.path.insert(0, str(STATS_DIR))
from category_resampling import block_bootstrap

METRICS = ('kept', 'accuracy', 'share')
IDENTITIES = ('model_state_hash', 'final_validation_context_axis_hash',
              'final_validation_context_design_hash')


def scalar(value):
    """A finite number from a receipt field."""
    if isinstance(value, bool): raise ValueError('boolean in numerical field')
    value = float(value)
    if not math.isfinite(value): raise ValueError('nonfinite numerical field')
    return value


def normalize(r):
    """A circuit's receipt as the compared measures: kept fraction, corrected accuracy (dense and circuit, in points),
    accuracy drop, held-out losses and their increase, the training loss span and the loss share (increase / span)."""
    cat = scalar(r['target_class_id'])
    if int(cat) != cat or not 0 <= cat < 1000: raise ValueError('invalid category')
    if int(r['query_id']) != cat: raise ValueError('category/query mismatch')
    kept = scalar(r['retained_fraction'])
    if not 0 <= kept <= 1: raise ValueError('invalid kept fraction')
    if not math.isclose(kept * r['route_count'], r['retained_route_count'], abs_tol=1e-6):
        raise ValueError('inconsistent retained weight count')
    d, f = r['terminal_dense_bce'], r['terminal_final_bce']
    da, fa = scalar(d['class_balanced_binary_accuracy']), scalar(f['class_balanced_binary_accuracy'])
    if not 0 <= da <= 1 or not 0 <= fa <= 1: raise ValueError('invalid accuracy')
    db, fb = scalar(d['class_balanced_binary_bce']), scalar(f['class_balanced_binary_bce'])
    if min(db,fb) < 0: raise ValueError('negative BCE')
    delta = scalar(r['terminal_bce_delta'])
    if not math.isclose(delta,fb-db,rel_tol=1e-9,abs_tol=1e-10):
        raise ValueError('inconsistent held-out loss delta')
    span = scalar(r['workflow']['loss_span_estimate'])
    if span <= 0 or math.log(2) <= db: raise ValueError('nonpositive span')
    result = dict(category=int(cat), tau=scalar(r['sensitivity_tau']), kept=kept,
                  accuracy=200*fa-100, dense_accuracy=200*da-100,
                  accuracy_drop=200*(da-fa), heldout_dense_bce=db,
                  heldout_final_bce=fb, heldout_delta_bce=delta,
                  training_span=span, share=delta/span)
    for key in IDENTITIES: result[key] = r[key]
    return result


def summarize(values, n_resamples=10000, seed=0, family_size=24):
    """For each measure (columns: kept, accuracy, share), the mean of the per-category differences with its percentile
    category bootstrap interval and the Bonferroni interval for a family of family_size."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(METRICS) or not np.isfinite(values).all():
        raise ValueError('invalid category matrix')
    result = {}
    for k, metric in enumerate(METRICS):
        item = block_bootstrap(values[:,k], n_resamples=n_resamples, seed=seed,
                               return_distribution=True)
        dist = item.pop('distribution')
        lo, hi = np.quantile(dist, [.05/(2*family_size), 1-.05/(2*family_size)])
        item.update(family_ci_low=float(lo), family_ci_high=float(hi))
        result[metric] = item
    return result


def dominance(contrasts, prefix='ci_'):
    """'dominates' when the intervals (prefix ci_ for marginal, family_ci_ for Bonferroni) show no more kept weights,
    higher accuracy and a lower loss share; otherwise 'unresolved or trade-off'."""
    if (contrasts['kept'][prefix+'high'] <= 0 and
        contrasts['accuracy'][prefix+'low'] > 0 and
        contrasts['share'][prefix+'high'] < 0): return 'dominates'
    return 'unresolved or trade-off'
