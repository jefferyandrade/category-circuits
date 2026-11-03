"""Category-tree structure of circuit activations against category prototypes at one layer (Fig. 7D; NumPy only).

The definitions are tree_share's, written out in NumPy: least-squares node effects, the share of pair similarity
explained by category-tree cells, and adjustment by 50 label permutations (seed 0; domain, superordinate and block50
in that order). Both representations are compared through their Pearson similarity between categories.

analyze_pair takes aligned category-by-feature means of the two representations (the circuits' activations and the
dense network's category means) and 1,000-output confusion counts. It returns, at each tree level, both tree shares,
their difference with a delete-one-category jackknife interval (t(.975, groups - 1) times the jackknife SE) and a
two-sided paired category-swap permutation test with the plus-one correction, and each representation's Spearman
correlation with the confusions. The values are JSON-compatible; undefined cells carry a reason and never a
floating-point substitute.
"""
import math
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


LEVELS = (("domain", "animate"), ("superordinate", "superordinate"), ("block50", "block50"))
TREE_PERMUTATIONS = 50
TREE_SEED = 0


def undefined(reason):
    return {"status": "undefined", "reason": str(reason)}


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _categories(categories):
    categories = list(categories)
    if (len(categories) < 4 or any(isinstance(x, (bool, np.bool_))
            or not isinstance(x, (int, np.integer)) or x < 0 or x >= 1000 for x in categories)
            or len(set(categories)) != len(categories)):
        raise ValueError("at least four unique integer category ids in [0,1000) required")
    return [int(x) for x in categories]


def _labels(tree, categories):
    """Accept aligned columns, category records, indexed mappings, or a DataFrame."""
    if hasattr(tree, "to_dict") and hasattr(tree, "columns"):
        if "category" in tree.columns:
            tree = tree.to_dict("records")
        else:
            tree = tree.to_dict("index")
    if isinstance(tree, (list, tuple)):
        tree = {int(row["category"]): row for row in tree}
    if not isinstance(tree, dict):
        raise ValueError("tree must supply category labels")
    result = {}
    for _, column in LEVELS:
        if column in tree:
            values = np.asarray(tree[column], dtype=object)
            if "category" in tree:
                index = {int(cat): i for i, cat in enumerate(tree["category"])}
                if len(index) != len(tree["category"]):
                    raise ValueError("tree category ids must be unique")
                try:
                    values = np.asarray([values[index[cat]] for cat in categories], dtype=object)
                except KeyError as error:
                    raise ValueError("categories absent from tree") from error
        else:
            try:
                values = np.asarray([(tree[cat] if cat in tree else tree[str(cat)])[column]
                                     for cat in categories], dtype=object)
            except (KeyError, TypeError) as error:
                raise ValueError("missing tree labels: " + column) from error
        if values.shape != (len(categories),) or any(x is None or
                isinstance(x, (float, np.floating)) and not np.isfinite(x) for x in values):
            raise ValueError("missing or misaligned tree labels: " + column)
        # Match tree_share's factorization of labels after conversion to strings.
        result[column] = np.asarray([str(x) for x in values], dtype=object)
    return result


def _rows(rows, n=None):
    raw = np.asarray(rows)
    if raw.ndim != 2 or raw.shape[1] < 2 or (n is not None and raw.shape[0] != n):
        raise ValueError("representations must be category rows with aligned features")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError("representations must be real finite numbers")
    if not np.isfinite(raw).all():
        raise ValueError("representations must be finite")
    precision = np.finfo(raw.dtype).eps if np.issubdtype(raw.dtype, np.floating) else np.finfo(float).eps
    return raw.astype(np.float64), float(precision)


def _normalized_rows(rows, n=None):
    x, precision = _rows(rows, n)
    scale = max(1., float(np.max(np.abs(x))))
    tolerance = 32 * precision * scale
    if np.max(np.abs(x - x[0])) <= tolerance:
        return None, "representations identical across categories at their numerical precision"
    centered = x - x.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    if np.any(norms <= tolerance * math.sqrt(x.shape[1])):
        return None, "one or more category representations have no feature variation"
    return centered / norms[:, None], None


def pearson_similarity(rows):
    normalized, reason = _normalized_rows(rows)
    if reason is not None:
        return dict(undefined(reason), similarity=None)
    similarity = np.clip(_gram(normalized, normalized), -1., 1.)
    np.fill_diagonal(similarity, 1.)
    return {"status": "formed", "similarity": similarity}


def _gram(left, right):
    # Some BLAS builds report stale floating-status flags; finite normalized
    # input and an explicit finite-output check bound errors.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = left @ right.T
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite normalized representation Gram matrix")
    return result


def _node_pairs(matrix):
    n = len(matrix)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (n, n) or n < 4:
        raise ValueError("node adjustment requires a square matrix of at least four categories")
    off = ~np.eye(n, dtype=bool)
    if not np.isfinite(matrix[off]).all() or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError("similarity must be symmetric with a finite off-diagonal")
    z = np.where(off, matrix, 0.)
    alpha = float(z.sum() / (n * (n - 1)))
    node = (z.sum(axis=1) - (n - 1) * alpha) / (n - 2)
    i, j = np.triu_indices(n, 1)
    e = matrix[i, j] - alpha - node[i] - node[j]
    if np.ptp(e) <= 32 * np.finfo(float).eps * max(1., float(np.max(np.abs(matrix[i, j])))):
        return None
    return e


class _TreePlan:
    """Reuse fixed label permutations for all paired category-row swaps."""
    def __init__(self, labels):
        n = len(labels["animate"])
        self.i, self.j = np.triu_indices(n, 1)
        self.levels = {}
        rng = np.random.default_rng(TREE_SEED)
        for level, column in LEVELS:
            unique, codes = {}, []
            for label in labels[column]:
                if label not in unique:
                    unique[label] = len(unique)
                codes.append(unique[label])
            codes, g = np.asarray(codes), len(unique)
            plans = []
            # Consume every RNG draw even if the level cannot be adjusted.
            for draw in range(TREE_PERMUTATIONS + 1):
                current = codes if draw == 0 else rng.permutation(codes)
                if g == 1 or g == n:
                    continue
                a, b = current[self.i], current[self.j]
                cell = (np.minimum(a, b) * g + np.maximum(a, b)).astype(np.int32)
                counts = np.bincount(cell, minlength=g * g)
                inverse = np.divide(1., counts, out=np.zeros(len(counts)), where=counts > 0)
                plans.append((cell, inverse))
            self.levels[level] = (g, plans)

    def evaluate(self, matrix):
        e = _node_pairs(matrix)
        if e is None:
            return {level: dict(undefined("no variation after node adjustment"), level=level)
                    for level, _ in LEVELS}
        centered = e - e.mean()
        total = float(np.sum(centered * centered))
        result = {}
        for level, (g, plans) in self.levels.items():
            if g == 1:
                result[level] = {"status": "formed", "level": level, "n_groups": 1,
                                 "share": 0., "share_permuted": 0., "share_adjusted": 0.}
                continue
            if not plans:
                result[level] = dict(undefined("permutation adjustment has no identifiable tree cells"),
                                     level=level, n_groups=g)
                continue
            values = []
            for cell, inverse in plans:
                sums = np.bincount(cell, weights=centered, minlength=len(inverse))
                values.append(float(np.sum(sums * sums * inverse) / total))
            observed, permuted = values[0], float(np.mean(values[1:]))
            if 1 - permuted <= 32 * np.finfo(float).eps:
                result[level] = dict(undefined("permutation adjustment is undefined"), level=level, n_groups=g)
                continue
            result[level] = {"status": "formed", "level": level, "n_groups": g,
                             "share": observed, "share_permuted": permuted,
                             "share_adjusted": (observed - permuted) / (1 - permuted)}
        return result


def tree_shares(similarity, categories, tree):
    categories = _categories(categories)
    if similarity is None:
        return {level: undefined("Pearson similarity is undefined") for level, _ in LEVELS}
    if np.asarray(similarity).shape != (len(categories), len(categories)):
        raise ValueError("similarity and categories differ in shape")
    return _TreePlan(_labels(tree, categories)).evaluate(similarity)


def confusion_pairs(counts, categories):
    """Each category pair's confusion rate: the mean of its two directed top-1 rates, each over its category's own
    image count (every training image, so the counts differ between categories)."""
    categories = list(categories)
    counts = np.asarray(counts)
    if (not categories or len(set(categories)) != len(categories)
            or any(not isinstance(c, (int, np.integer)) or isinstance(c, (bool, np.bool_))
                   or c < 0 or c >= 1000 for c in categories)
            or counts.shape != (len(categories), 1000) or counts.dtype.kind not in 'iu'
            or (counts < 0).any()):
        raise ValueError('confusion counts must be aligned nonnegative integer 1000-output rows')
    totals = counts.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError('every confusion category must contain images')
    shared = counts[:, categories]
    directed = shared / totals[:, None]
    symmetric = (directed + directed.T) / 2
    np.fill_diagonal(symmetric, 0.)
    return dict(pair_rate=symmetric, outside_population_counts=totals - shared.sum(axis=1),
        image_counts=totals,
        normalization='(C_ij/n_i + C_ji/n_j)/2; unconditional full-image 1000-output counts')


def _ranks(values):
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start + stop + 1) / 2
    return ranks


def confusion_spearman(similarity, confusion):
    confusion = np.asarray(confusion, dtype=float)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion pair matrix must be square")
    n = len(confusion)
    result = {"n_pairs": n * (n - 1) // 2, "rho": None,
              "pair_selection": "all unordered off-diagonal category pairs, including zeros", "ties": "average ranks"}
    if similarity is None:
        return dict(result, **undefined("Pearson similarity is undefined"))
    similarity = np.asarray(similarity, dtype=float)
    if similarity.shape != confusion.shape:
        raise ValueError("similarity and confusion differ in shape")
    indices = np.triu_indices(n, 1)
    a, b = similarity[indices], confusion[indices]
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(result, **undefined("requires at least two finite category pairs"))
    a, b = _ranks(a), _ranks(b)
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denominator == 0:
        return dict(result, **undefined("similarity or confusion ranks are constant"))
    return dict(result, status="formed", rho=float(np.clip(np.sum(a * b) / denominator, -1., 1.)))


def jackknife_groups(n_categories, seed=0):
    if n_categories <= 200:
        return [np.asarray([i], dtype=int) for i in range(n_categories)]
    return list(np.array_split(np.random.default_rng(seed).permutation(n_categories), 100))


def _beta_fraction(a, b, x):
    """Continued fraction for regularized incomplete beta, stdlib only."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1, a - 1
    c, d = 1., 1 - qab * x / qap
    d = 1 / (d if abs(d) >= tiny else tiny)
    h = d
    for m in range(1, 1001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / (d if abs(d) >= tiny else tiny)
        c = c if abs(c) >= tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / (d if abs(d) >= tiny else tiny)
        c = c if abs(c) >= tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-15:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.
    if x >= 1:
        return 1.
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _beta_fraction(a, b, x) / a
    return 1 - front * _beta_fraction(b, a, 1 - x) / b


def student_t_critical(df):
    """Student t .975 quantile, as used in category_resampling' category jackknife."""
    if df < 1:
        raise ValueError("positive degrees of freedom required")
    lo, hi = 0., 128.
    for _ in range(65):
        mid = (lo + hi) / 2
        cdf = 1 - .5 * _regularized_beta(df / (df + mid * mid), df / 2, .5)
        if cdf < .975:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _difference(first, second, level):
    a, b = first[level], second[level]
    if a["status"] != "formed" or b["status"] != "formed":
        return None
    return float(a["share_adjusted"] - b["share_adjusted"])


def analyze_pair(computation, image, categories, tree, confusion_counts, second=None,
                 seed=0, permutations=1000):
    """Every measure for aligned category representation means.

    computation is the circuits' activations, image the prototypes and second the P-redraw circuits' activations;
    the results keep these names. Paired swaps exchange a category's entire feature vector between the two
    representations before the Pearson similarities are recomputed; tests are two-sided with the plus-one correction.
    The jackknife deletes single categories at up to 200 categories (fixed random groups beyond), and its t(.975)
    intervals match the category jackknife of category_resampling. Tree permutations keep seed 0 independently of the swap seed. No undefined resample is
    discarded.
    """
    categories = _categories(categories)
    n = len(categories)
    labels = _labels(tree, categories)
    a, _ = _rows(computation, n)
    b, _ = _rows(image, n)
    if a.shape != b.shape:
        raise ValueError("computation and image must have the same category rows and features")
    if isinstance(permutations, bool) or not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError("a positive integer number of permutations is required")
    confusion = confusion_pairs(confusion_counts, categories)
    plan = _TreePlan(labels)
    families, similarities = {}, {}
    for name, rows in (("computation", computation), ("image", image), ("second", second)):
        if rows is None:
            continue
        raw, _ = _rows(rows, n)
        if raw.shape != a.shape:
            raise ValueError("second-run category rows and features must match")
        pearson = pearson_similarity(rows)
        similarity = pearson["similarity"]
        similarities[name] = similarity
        shares = plan.evaluate(similarity) if similarity is not None else {
            level: undefined(pearson["reason"]) for level, _ in LEVELS}
        families[name] = {"pearson_status": pearson["status"], "pearson_reason": pearson.get("reason"),
                          "pearson_similarity": similarity, "tree": shares,
                          "confusion_spearman": confusion_spearman(similarity, confusion["pair_rate"])}
    groups = jackknife_groups(n, seed)
    output = {"n_categories": n, "categories": categories, "feature_dimension": a.shape[1],
              "families": families, "confusion": confusion, "computation_minus_image": {},
              "first_minus_second": {}, "settings": {
                  "tree_adjustment": "node and permutation adjusted", "tree_permutations": TREE_PERMUTATIONS,
                  "tree_seed": TREE_SEED, "swap_seed": int(seed), "swap_permutations": int(permutations),
                  "jackknife_seed": int(seed), "jackknife_n_groups": len(groups),
                  "jackknife_groups": [[categories[i] for i in group] for group in groups],
                  "jackknife_unit": "category" if n <= 200 else "fixed random category group",
                  "confidence_interval": "estimate +/- Student t(.975, n_groups-1) * jackknife SE",
                  "permutation_test": "paired category representation-row swap; two-sided; plus-one correction"}}
    observed = {}
    for level, _ in LEVELS:
        diff = _difference(families["computation"]["tree"], families["image"]["tree"], level)
        observed[level] = diff
        if diff is None:
            cell = dict(undefined("one or both tree shares are undefined"), difference=None,
                        jackknife=undefined("observed difference is undefined"),
                        permutation=undefined("observed difference is undefined"))
        else:
            cell = {"status": "formed", "difference": diff}
        output["computation_minus_image"][level] = cell
        if second is not None:
            redraw = _difference(families["computation"]["tree"], families["second"]["tree"], level)
            output["first_minus_second"][level] = ({"status": "formed", "difference": redraw}
                if redraw is not None else dict(undefined("one or both tree shares are undefined"), difference=None))
    active = [level for level in observed if observed[level] is not None]
    if not active:
        return _clean(output)
    deleted = {level: [] for level in active}
    for group in groups:
        keep = np.ones(n, dtype=bool)
        keep[group] = False
        if keep.sum() < 4:
            for level in active:
                deleted[level].append(None)
            continue
        subset = np.ix_(keep, keep)
        deletion_plan = _TreePlan({column: values[keep] for column, values in labels.items()})
        left = deletion_plan.evaluate(similarities["computation"][subset])
        right = deletion_plan.evaluate(similarities["image"][subset])
        for level in active:
            deleted[level].append(_difference(left, right, level))
    critical = student_t_critical(len(groups) - 1)
    for level in active:
        values = deleted[level]
        if any(x is None for x in values):
            summary = dict(undefined("at least one category deletion has undefined tree share"),
                           n_groups=len(groups), deleted_differences=values)
        else:
            values = np.asarray(values)
            se = math.sqrt(float((len(groups) - 1) / len(groups) * np.sum((values - values.mean()) ** 2)))
            summary = {"status": "formed", "estimate": observed[level], "se": se,
                       "ci_low": observed[level] - critical * se, "ci_high": observed[level] + critical * se,
                       "critical_value": critical, "n_groups": len(groups), "deleted_differences": values}
        output["computation_minus_image"][level]["jackknife"] = summary
    # Precompute all four source-row Gram combinations once. Each draw costs
    # two O(n^2) selections plus fixed-label pair-cell sums, independent of D.
    na, _ = _normalized_rows(computation)
    nb, _ = _normalized_rows(image)
    aa, bb, ab = _gram(na, na), _gram(nb, nb), _gram(na, nb)
    rng = np.random.default_rng(seed)
    null = {level: [] for level in active}
    for _ in range(int(permutations)):
        swap = rng.integers(0, 2, n).astype(bool)
        si, sj = swap[:, None], swap[None, :]
        left_matrix = np.where(si, np.where(sj, bb, ab.T), np.where(sj, ab, aa))
        right_matrix = np.where(si, np.where(sj, aa, ab), np.where(sj, ab.T, bb))
        left, right = plan.evaluate(left_matrix), plan.evaluate(right_matrix)
        for level in active:
            null[level].append(_difference(left, right, level))
    for level in active:
        values = null[level]
        if any(x is None for x in values):
            summary = dict(undefined("at least one paired swap has undefined tree share"),
                           permutations=int(permutations), null_differences=values)
        else:
            tolerance = 64 * np.finfo(float).eps * max(1., abs(observed[level]))
            exceedances = sum(abs(x) >= abs(observed[level]) - tolerance for x in values)
            summary = {"status": "formed", "permutations": int(permutations), "seed": int(seed),
                       "exceedances": exceedances, "p_two_sided": (1 + exceedances) / (1 + int(permutations)),
                       "null_differences": values}
        output["computation_minus_image"][level]["permutation"] = summary
    return _clean(output)
