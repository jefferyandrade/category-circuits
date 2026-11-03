"""Statistics of the AlexNet fc6 and fc7 comparisons (Fig. 7B; the AlexNet fc6 and fc7 points of Fig. 7D).

Pearson similarity between categories; node- and permutation-adjusted category-tree shares, with the algebra of
tree_share.node_adjust and tree_rows, its 50 label permutations and their order of random draws; the Spearman
correlation of a similarity with the confusions; and, for the difference between two representations, a
delete-category jackknife (single categories, or fixed random groups of ten at all 1,000 categories and in the
populations drawn from them) and a paired category-swap permutation test. Label cells are prepared once for a
population, and category swaps select four precomputed cross-Grams. No category pair is treated as an independent
sampling unit. json_clean and category_ids serve fc_all_images.py and whole_circuits.py.
"""
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stats"))
import tree_share as TS  # noqa: E402


TREE_LEVELS = (("domain", "animate"), ("superordinate", "superordinate"), ("block50", "block50"))
METRICS = tuple("tree_" + level for level, _ in TREE_LEVELS) + ("confusion_rho",)
REL_TOL = 1e-10   # a centered row below this fraction of the input's scale is roundoff, not variation


def undefined(reason):
    return {"status": "undefined", "reason": str(reason)}


def _finite(value):
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)


def json_clean(value):
    """The value with arrays as lists and non-finite floats as None, ready for json.dump."""
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def category_ids(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(name + " must be a nonempty list of integer category ids")
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in values):
        raise ValueError(name + " must contain nonnegative integer category ids")
    if len(set(values)) != len(values) or values != sorted(values):
        raise ValueError(name + " must be unique and sorted")
    return values


def centered_scaled(m):
    """Rows centered, then scaled by their largest entry (no overflow); a row that is roundoff relative to the input's
    own scale is undefined."""
    m = np.asarray(m, dtype=np.float64)
    if not np.isfinite(m).all():
        raise ValueError("non-finite row")
    c = m - m.mean(1, keepdims=True)
    s = np.abs(c).max(1, keepdims=True)
    if (s <= REL_TOL * max(np.abs(m).max(), np.finfo(float).tiny)).any():
        raise ValueError("undefined: a constant row (variation at roundoff)")
    return c / s


def normalized_rows(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("representations must be a finite category-by-feature matrix")
    centered = centered_scaled(values)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms == 0):
        raise ValueError("Pearson similarity has a constant feature vector")
    return centered / norms[:, None]


def category_invariant(vectors):
    """Identical category responses have no category geometry, even with roundoff."""
    values = np.asarray(vectors, dtype=np.float64)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    return bool(np.max(np.abs(values - values[:1])) <= REL_TOL * scale)


def pearson_similarity(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    rows = normalized_rows(values)
    if category_invariant(values):
        raise ValueError("representation is identical across categories")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = rows @ rows.T
    if not np.isfinite(result).all():
        raise ValueError("non-finite normalized Pearson Gram")
    np.clip(result, -1., 1., out=result)
    np.fill_diagonal(result, 1.)
    return result


class TreePlan:
    """Fixed taxonomy cell indices, preserving tree_share's permutation stream.

    At n=1,000 the int32 cells occupy approximately 306 MB (153 x 499,500).
    A plan is reused for every swap and discarded before the next layer.
    """
    def __init__(self, categories, taxonomy):
        self.categories = np.asarray(categories, dtype=np.int64)
        n = len(self.categories)
        if n < 4 or len(np.unique(self.categories)) != n:
            raise ValueError("tree share requires at least four unique categories")
        tax = taxonomy.set_index("category") if "category" in taxonomy else taxonomy
        if not tax.index.is_unique or not np.isin(self.categories, tax.index).all():
            raise ValueError("taxonomy must uniquely label every category")
        self.iu = np.triu_indices(n, 1)
        rng = np.random.default_rng(0)
        self.levels = {}
        for level, column in TREE_LEVELS:
            if column not in tax or tax.loc[self.categories, column].isna().any():
                raise ValueError("missing taxonomy labels: " + column)
            codes = pd.factorize(tax.loc[self.categories, column].astype(str).values)[0]
            groups = int(codes.max()) + 1
            cells = []
            for iteration in range(51):
                labels = codes if iteration == 0 else rng.permutation(codes)
                cell = TS.cell_index(labels, self.iu, groups).astype(np.int32)
                counts = np.bincount(cell, minlength=groups * groups)
                cells.append((cell, counts))
            self.levels[level] = (groups, codes, cells)

    def evaluate(self, similarity):
        similarity = np.asarray(similarity, dtype=np.float64)
        n = len(self.categories)
        if similarity.shape != (n, n) or not np.isfinite(similarity).all():
            raise ValueError("similarity must be finite and align with categories")
        if not np.allclose(similarity, similarity.T, rtol=0., atol=1e-12):
            raise ValueError("similarity must be symmetric")
        residual = TS.node_adjust(similarity)[self.iu]
        tolerance = np.finfo(float).eps * max(1., float(np.max(np.abs(similarity[self.iu]))))
        if np.ptp(residual) <= tolerance:
            return {level: undefined("no variation after node adjustment") for level, _ in TREE_LEVELS}
        grand = residual.mean()
        denominator = np.sum((residual - grand) ** 2)
        result = {}
        for level, (groups, codes, cells) in self.levels.items():
            shares = []
            for cell, counts in cells:
                sums = np.bincount(cell, weights=residual, minlength=len(counts))
                means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.)
                shares.append(float(np.sum(counts * (means - grand) ** 2) / denominator))
            share, permuted = shares[0], float(np.mean(shares[1:]))
            if abs(1. - permuted) <= np.finfo(float).eps:
                result[level] = undefined("permutation adjustment is undefined")
                continue
            within = codes[self.iu[0]] == codes[self.iu[1]]
            contrast = float(residual[within].mean() - residual[~within].mean()) if within.any() and (~within).any() else None
            result[level] = {"status": "formed", "level": level, "n_groups": groups,
                             "share": share, "share_permuted": permuted,
                             "share_adjusted": (share - permuted) / (1. - permuted),
                             "contrast_within_minus_between": contrast}
        return result


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
    return ranks


def confusion_spearman(similarity, confusion):
    """Spearman correlation of the pair similarities with the pair confusion rates: the Pearson correlation of their
    average ranks over every unordered category pair, zero-confusion pairs included."""
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion pair matrix must be square")
    n = len(confusion)
    result = {"n_pairs": n * (n - 1) // 2,
              "pair_selection": "all unordered off-diagonal category pairs, including zeros",
              "ties": "average ranks", "rho": None}
    if similarity is None:
        return dict(result, **undefined("Pearson similarity is undefined"))
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.shape != confusion.shape:
        raise ValueError("similarity and confusion pair matrices differ in shape")
    indices = np.triu_indices(n, 1)
    a, b = similarity[indices], confusion[indices]
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(result, **undefined("requires at least two finite category pairs"))
    if np.ptp(a) <= REL_TOL * max(1., float(np.abs(a).max())):
        return dict(result, **undefined("similarity is constant (variation at roundoff)"))
    a, b = _average_ranks(a), _average_ranks(b)
    a, b = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return dict(result, **undefined("similarity or confusion ranks are constant"))
    return dict(result, status="formed", rho=float(np.clip(a @ b / norm, -1, 1)))


def similarity_metrics(similarity, confusion, categories, taxonomy, plan=None):
    if similarity is None:
        tree = {level: undefined("category geometry is undefined") for level, _ in TREE_LEVELS}
    elif len(categories) < 4:
        tree = {level: undefined("tree share requires at least four categories") for level, _ in TREE_LEVELS}
    else:
        tree = (plan or TreePlan(categories, taxonomy)).evaluate(similarity)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rho = confusion_spearman(similarity, confusion)
    return {"tree": tree, "confusion_spearman": rho}


def representation_metrics(vectors, confusion, categories, taxonomy, plan=None):
    try:
        similarity = pearson_similarity(vectors)
        reason = None
    except ValueError as error:
        similarity, reason = None, str(error)
    result = similarity_metrics(similarity, confusion, categories, taxonomy, plan)
    result["pearson_similarity"] = similarity
    if reason:
        result["undefined_reason"] = reason
    return result


def metric_values(metrics):
    result = {"tree_" + level: metrics["tree"][level].get("share_adjusted", np.nan)
              for level, _ in TREE_LEVELS}
    result["confusion_rho"] = metrics["confusion_spearman"].get("rho", np.nan)
    return {name: float(value) if _finite(value) else np.nan for name, value in result.items()}


def jackknife_groups(categories, seed=0, groups_of_ten=False):
    """Fixed random groups of ten for all 1,000 categories, and with groups_of_ten for any population: a seed-0 random
    order cut by np.array_split into ceil(n / 10) nearly equal groups (the 900 held-out categories make 90 groups of
    ten); single-category deletions otherwise."""
    categories = np.asarray(categories, dtype=np.int64)
    if groups_of_ten or len(categories) == 1000:
        order = np.random.default_rng(seed).permutation(len(categories))
        return [np.sort(group) for group in np.array_split(order, math.ceil(len(categories) / 10))]
    return [np.asarray([index]) for index in range(len(categories))]


def jackknife_summary(full, replicates):
    replicates = np.asarray(replicates, dtype=np.float64)
    if not _finite(full) or len(replicates) < 2 or not np.isfinite(replicates).all():
        return dict(undefined("full value or a deletion replicate is undefined"), difference=full,
                    n_replicates=len(replicates), n_defined_replicates=int(np.isfinite(replicates).sum()))
    groups = len(replicates)
    se = math.sqrt((groups - 1.) / groups * np.sum((replicates - replicates.mean()) ** 2))
    return {"status": "formed", "difference": float(full), "se": se,
            "lo": float(full - 1.96 * se), "hi": float(full + 1.96 * se),
            "confidence": .95, "critical_value": 1.96, "n_replicates": groups,
            "replicates": replicates.tolist()}


def swap_similarity_grams(left, right):
    """Four within/cross Grams let each swap cost O(n squared), not O(n squared d)."""
    a, b = normalized_rows(left), normalized_rows(right)
    if a.shape != b.shape:
        raise ValueError("paired representations must have identical category/feature dimensions")
    # Some BLAS builds leave floating status flags set despite finite bounded
    # dot products. Check returned values explicitly rather than their flags.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        aa, ab, bb = a @ a.T, a @ b.T, b @ b.T
    if not all(np.isfinite(matrix).all() for matrix in (aa, ab, bb)):
        raise ValueError("non-finite normalized cross-Gram")
    return aa, ab, ab.T, bb


def swapped_similarities(grams, swap):
    swap = np.asarray(swap, dtype=bool)
    aa, ab, ba, bb = grams
    row, col = swap[:, None], swap[None, :]
    left = np.where(row, np.where(col, bb, ba), np.where(col, ab, aa))
    right = np.where(row, np.where(col, aa, ab), np.where(col, ba, bb))
    np.clip(left, -1., 1., out=left)
    np.clip(right, -1., 1., out=right)
    np.fill_diagonal(left, 1.)
    np.fill_diagonal(right, 1.)
    return left, right


def permutation_summary(full, replicates, seed=0):
    replicates = np.asarray(replicates, dtype=np.float64)
    common = {"n_permutations": len(replicates), "seed": seed,
              "tail": "two-sided absolute difference; inclusive ties; plus-one correction",
              "n_defined_permutations": int(np.isfinite(replicates).sum())}
    if not _finite(full) or not np.isfinite(replicates).all():
        return dict(common, **undefined("observed difference or a category-swap replicate is undefined"))
    # A tolerance only covers arithmetic roundoff in complementary swaps.
    tolerance = 32 * np.finfo(float).eps * max(1., abs(full))
    extreme = int(np.sum(np.abs(replicates) >= abs(full) - tolerance))
    return dict(common, status="formed", difference=float(full), n_extreme=extreme,
                p_two_sided=(extreme + 1.) / (len(replicates) + 1.),
                null_differences=replicates.tolist())


def paired_difference(left, right, categories, taxonomy, confusion,
                      n_permutations=1000, seed=0, group_seed=0, plan=None, groups_of_ten=False):
    """Joint deletion and paired category-vector swap for all four statistics (three tree shares and the confusion
    correlation) of left minus right. The deletions are jackknife_groups(categories, group_seed, groups_of_ten)."""
    left, right = np.asarray(left), np.asarray(right)
    categories = np.asarray(categories, dtype=np.int64)
    confusion = np.asarray(confusion, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or len(left) != len(categories):
        raise ValueError("paired representations must align with categories and feature dimensions")
    if confusion.shape != (len(categories), len(categories)):
        raise ValueError("confusions must align with categories")
    if n_permutations < 1:
        raise ValueError("at least one category-swap permutation is required")
    plan = plan or (TreePlan(categories, taxonomy) if len(categories) >= 4 else None)
    left_metrics = representation_metrics(left, confusion, categories, taxonomy, plan)
    right_metrics = representation_metrics(right, confusion, categories, taxonomy, plan)
    left_sim, right_sim = left_metrics["pearson_similarity"], right_metrics["pearson_similarity"]
    lm, rm = metric_values(left_metrics), metric_values(right_metrics)
    full = {metric: lm[metric] - rm[metric] for metric in METRICS}
    groups = jackknife_groups(categories, group_seed, groups_of_ten)
    jk = {metric: [] for metric in METRICS}
    for deleted in groups:
        keep = np.ones(len(categories), dtype=bool)
        keep[deleted] = False
        ids = np.flatnonzero(keep)
        local_confusion = confusion[np.ix_(ids, ids)]
        local_plan = TreePlan(categories[keep], taxonomy) if keep.sum() >= 4 else None
        a_sim = None if left_sim is None or category_invariant(left[keep]) else left_sim[np.ix_(ids, ids)]
        b_sim = None if right_sim is None or category_invariant(right[keep]) else right_sim[np.ix_(ids, ids)]
        a = metric_values(similarity_metrics(a_sim, local_confusion, categories[keep], taxonomy, local_plan))
        b = metric_values(similarity_metrics(b_sim, local_confusion, categories[keep], taxonomy, local_plan))
        for metric in METRICS:
            jk[metric].append(a[metric] - b[metric])
    random = {metric: [] for metric in METRICS}
    try:
        grams = swap_similarity_grams(left, right)
    except ValueError:
        grams = None
    rng = np.random.default_rng(seed)
    for _ in range(n_permutations):
        swap = rng.random(len(categories)) < .5
        if grams is None:
            a = b = {metric: np.nan for metric in METRICS}
        else:
            a_sim, b_sim = swapped_similarities(grams, swap)
            # Identical Pearson rows can leave last-bit off-diagonal variation.
            iu = np.triu_indices(len(categories), 1)
            a_sim = None if np.ptp(a_sim[iu]) <= 32 * np.finfo(float).eps else a_sim
            b_sim = None if np.ptp(b_sim[iu]) <= 32 * np.finfo(float).eps else b_sim
            a = metric_values(similarity_metrics(a_sim, confusion, categories, taxonomy, plan))
            b = metric_values(similarity_metrics(b_sim, confusion, categories, taxonomy, plan))
        for metric in METRICS:
            random[metric].append(a[metric] - b[metric])
    sizes = sorted({len(group) for group in groups})
    return {"n_categories": len(categories),
            "jackknife_unit": ("%d fixed random groups of %s categories" % (len(groups), " or ".join(map(str, sizes)))
                               if max(sizes, default=1) > 1 else "category"),
            "group_seed": group_seed,
            "deleted_category_groups": [categories[group].tolist() for group in groups],
            "metrics": {metric: {"left": lm[metric], "right": rm[metric], "difference": full[metric],
                                  "jackknife": jackknife_summary(full[metric], jk[metric]),
                                  "category_swap": permutation_summary(full[metric], random[metric], seed)}
                        for metric in METRICS}}
