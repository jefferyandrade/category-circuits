"""Input divergence of category pairs and its references (Fig. 4C), from integer pair sums.

At a unit with n inputs, two circuits that keep c1 and c2 of them, I in common, diverge by min(c1, c2) - I out of a
possible min(c1, c2) - max(0, c1 + c2 - n). A pair's input divergence is the summed divergence over the units divided
by the summed maximum; units where the maximum is zero (a circuit keeps none or all of the unit's inputs) add zero to
both sums. The references replace the shared count I by its expectation if each circuit chose its kept inputs at
random, among all of the unit's inputs (uniform) or within each decile of input weight size (weight-size-matched):
E[I] = sum over strata of c1_s c2_s / n_s. Pair sums are exact integers (binary64 for the references); a layer's
value is the mean over the pairs whose maximum is positive, and its interval a delete-one-category jackknife,
point +/- t(0.975, Q - 1) x SE, deleting every category in turn.
"""
import numpy as np


def decile_labels(scores):
    """Ten rank deciles of each unit's inputs by weight size: floor(10 r / n) with ties in input order;
    a unit with fewer than ten inputs has one stratum."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] == 0 or not np.isfinite(scores).all():
        raise ValueError('input scores must be a finite unit-by-input array')
    units, inputs = scores.shape
    labels = np.zeros((units, inputs), dtype=np.uint8)
    if inputs >= 10:
        order = np.argsort(scores, axis=1, kind='stable')
        ranks = (10 * np.arange(inputs) // inputs).astype(np.uint8)
        np.put_along_axis(labels, order, np.broadcast_to(ranks, labels.shape), axis=1)
    return labels


def reference_totals(counts, strata, sizes, return_all=False):
    """Pair sums of the bound min(c1, c2), the uniform and weight-size-matched references and the
    maximum divergence, from kept counts per unit (and per unit and stratum). Units where a circuit keeps none or
    all inputs contribute equally to the bound and the expected overlap, so they cancel."""
    q, units = counts.shape
    inputs = sizes.sum(axis=1)
    low = np.zeros((q, q), dtype=np.int64)
    den = np.zeros_like(low)
    for unit in range(units):
        a = counts[:, unit].astype(np.int64)
        mn = np.minimum(a[:, None], a[None, :])
        low += mn
        den += mn - np.maximum(0, a[:, None] + a[None, :] - inputs[unit])
    z = counts.astype(np.float64) / np.sqrt(inputs)[None, :]
    uniform = low - z @ z.T
    z = np.divide(strata, np.sqrt(sizes)[None, :, :],
                  out=np.zeros(strata.shape, dtype=np.float64),
                  where=sizes[None, :, :] > 0).reshape(q, -1)
    matched = low - z @ z.T
    if return_all:
        return low, uniform, matched, den
    return low, uniform, matched


class PairTotals:
    """Accumulate pair sums over batches of units.

    A mask is boolean [category, unit, input], a convolution's inputs being its input channels. The observed
    intersections are exact integer Gram products; category_ids, when given, catch a reordered category axis."""

    def __init__(self, n_categories, category_ids=None):
        if int(n_categories) != n_categories or n_categories < 2:
            raise ValueError('at least two categories are required')
        self.n_categories = int(n_categories)
        shape = (self.n_categories, self.n_categories)
        self.numerator = np.zeros(shape, dtype=np.int64)
        self.denominator = np.zeros(shape, dtype=np.int64)
        self.uniform_numerator = np.zeros(shape, dtype=np.float64)
        self.weight_numerator = np.zeros(shape, dtype=np.float64)
        self.partial_units = np.zeros(shape, dtype=np.int64)
        self.n_units = 0
        self.category_ids = None
        if category_ids is not None:
            self.category_ids = self._category_ids(category_ids)

    def _category_ids(self, category_ids):
        ids = tuple(category_ids)
        if len(ids) != self.n_categories or len(set(ids)) != self.n_categories:
            raise ValueError('category ids must be unique and match category count')
        return ids

    def update(self, mask, strata, input_chunk=8192, category_ids=None):
        mask, strata = np.asarray(mask), np.asarray(strata)
        if mask.ndim != 3 or mask.dtype != np.bool_ or mask.shape[0] != self.n_categories:
            raise ValueError('mask must be boolean [category,unit,input]')
        q, units, inputs = mask.shape
        if inputs == 0 or units == 0:
            raise ValueError('unit and input dimensions must be nonempty')
        if (strata.shape != (units, inputs) or not np.issubdtype(strata.dtype, np.integer)
                or np.any(strata < 0) or np.any(strata > 9)):
            raise ValueError('strata must be unit-by-input integer labels from 0 to 9')
        if int(input_chunk) != input_chunk or input_chunk <= 0:
            raise ValueError('input_chunk must be positive')
        if category_ids is not None:
            ids = self._category_ids(category_ids)
            if self.category_ids is not None and ids != self.category_ids:
                raise ValueError('category order changed between unit batches')
            self.category_ids = ids
        counts = mask.sum(axis=2, dtype=np.int64)
        n_strata = int(strata.max()) + 1
        sizes = np.zeros((units, n_strata), dtype=np.int64)
        stratum_counts = np.zeros((q, units, n_strata), dtype=np.int64)
        for stratum in range(n_strata):
            membership = strata == stratum
            sizes[:, stratum] = membership.sum(axis=1)
            stratum_counts[:, :, stratum] = (mask & membership[None, :, :]).sum(axis=2, dtype=np.int64)
        low, uniform, matched, den = reference_totals(counts, stratum_counts, sizes, return_all=True)
        flat = mask.reshape(q, -1)
        intersection = np.zeros((q, q), dtype=np.float64)
        for start in range(0, flat.shape[1], int(input_chunk)):
            part = flat[:, start:start + int(input_chunk)].astype(np.float64)
            intersection += part @ part.T
        # Each Gram sum is an exact nonnegative integer below 2**53; check before converting.
        if flat.shape[1] >= 2 ** 53 or not np.equal(intersection, np.rint(intersection)).all():
            raise ValueError('observed intersection did not remain exact integer arithmetic')
        partial = ((counts > 0) & (counts < inputs)).astype(np.float64)
        self.partial_units += np.rint(partial @ partial.T).astype(np.int64)
        self.numerator += low - intersection.astype(np.int64)
        self.denominator += den
        self.uniform_numerator += uniform
        self.weight_numerator += matched
        self.n_units += units

    def finalize(self):
        if self.n_units == 0:
            raise ValueError('no unit batches have been accumulated')
        result = {name: getattr(self, name).copy() for name in
                  ('numerator', 'denominator', 'uniform_numerator',
                   'weight_numerator', 'partial_units')}
        for name in result:
            np.fill_diagonal(result[name], 0)
        result['n_units'] = self.n_units
        result['n_categories'] = self.n_categories
        return result


def pair_values(num, den):
    """Pair ratios num / den; undefined pairs (den = 0) and the diagonal are NaN."""
    result = np.full(den.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=result, where=den > 0)
    np.fill_diagonal(result, np.nan)
    return result


def _critical(n_categories, critical):
    """t(0.975, Q - 1) unless a critical value is given."""
    if critical is None:
        from scipy.stats import t
        critical = float(t.ppf(.975, n_categories - 1))
    if not np.isfinite(critical) or critical <= 0:
        raise ValueError('critical value must be finite and positive')
    return float(critical)


def _mean_delete(x):
    """Mean over the defined pairs, and every delete-one-category mean from pair totals and row sums."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or len(x) < 2:
        raise ValueError('expected a square category-pair matrix')
    q = len(x)
    valid = np.isfinite(x)
    np.fill_diagonal(valid, False)
    iu = np.triu_indices(q, 1)
    n_pairs = int(valid[iu].sum())
    total = float(np.where(valid[iu], x[iu], 0.).sum())
    row_sum = np.where(valid, x, 0.).sum(axis=1)
    row_n = valid.sum(axis=1)
    deleted = np.full(q, np.nan)
    np.divide(total - row_sum, n_pairs - row_n, out=deleted, where=n_pairs > row_n)
    return total / n_pairs if n_pairs else float('nan'), deleted, n_pairs


def _interval(point, deleted, critical):
    """Jackknife standard error and interval; missing when any deletion is undefined."""
    q = len(deleted)
    n_defined = int(np.isfinite(deleted).sum())
    if n_defined != q:
        return float('nan'), float('nan'), float('nan'), n_defined
    se = float(np.sqrt((q - 1.) / q * np.square(deleted - deleted.mean()).sum()))
    return se, point - critical * se, point + critical * se, n_defined


def jackknife(x, critical=None):
    """Mean over defined pairs with a delete-one-category jackknife interval, deleting every category
    (including categories without a defined pair)."""
    point, deleted, n_pairs = _mean_delete(x)
    q = len(x)
    crit = _critical(q, critical)
    se, lower, upper, n_defined = _interval(point, deleted, crit)
    return dict(point=point, se=se, lower=lower, upper=upper, critical=crit,
                n_categories=q, n_pairs=n_pairs, n_delete_defined=n_defined)


def summarize(totals, critical=None):
    """A layer's values: observed divergence, both references and the differences with their intervals,
    all over the same defined pairs."""
    den = totals['denominator']
    observed = pair_values(totals['numerator'], den)
    uniform = pair_values(totals['uniform_numerator'], den)
    weight = pair_values(totals['weight_numerator'], den)
    obs_summary = jackknife(observed, critical=critical)
    difference_uniform = jackknife(observed - uniform, critical=critical)
    difference_weight = jackknife(observed - weight, critical=critical)
    iu = np.triu_indices(len(den), 1)
    valid = den[iu] > 0
    n_pairs = len(iu[0])
    share = totals['partial_units'][iu].astype(np.float64) / totals['n_units']
    result = dict(observed=obs_summary['point'], ci_low=obs_summary['lower'], ci_high=obs_summary['upper'],
                  ref_uniform=float(uniform[iu][valid].mean()) if valid.any() else float('nan'),
                  ref_weight=float(weight[iu][valid].mean()) if valid.any() else float('nan'),
                  obs_minus_uniform=difference_uniform['point'],
                  obs_minus_uniform_ci_low=difference_uniform['lower'],
                  obs_minus_uniform_ci_high=difference_uniform['upper'],
                  obs_minus_weight=difference_weight['point'],
                  obs_minus_weight_ci_low=difference_weight['lower'],
                  obs_minus_weight_ci_high=difference_weight['upper'],
                  n_pairs_defined=int(valid.sum()), n_pairs=n_pairs,
                  mean_partial_share=float(share.mean()),
                  mean_partial_share_defined=float(share[valid].mean()) if valid.any() else float('nan'),
                  n_delete_defined=obs_summary['n_delete_defined'],
                  flagged_omit=bool(int(valid.sum()) < n_pairs / 2.),
                  n_categories=len(den))
    return result
