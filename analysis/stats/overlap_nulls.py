"""Fig. 4B and 4D: the circuits' overlap at one layer against the size-, sign- and weight-size-matched references.

One weight layer per call. Inputs: the layer's gram (I_all: the exact number of kept weights each pair of circuits
shares, for the P circuits followed by their repeat extractions P-redraw where those exist; its diagonal holds the
kept counts), the circuits' masks (read again and checked against their receipts, to count each circuit's kept
weights in each weight stratum and how many circuits keep each weight, t_r) and the checkpoint's weights.

Strata, from the checkpoint alone: sign x decile of |w| within the sign, 20 per layer (analysis/common/strata.py).
Three references for a pair of circuits that keep a and b weights:
  uniform    both sets placed at random in the layer (the size-matched reference J0),
  sign       placed at random within each sign, with the circuits' own counts per sign,
  magnitude  placed at random within each of the 20 strata, with the circuits' own counts per stratum (the sign- and
             weight-size-matched reference).
Under each reference the intersection X is a sum of independent hypergeometric counts. For pairs of P circuits the
uniform J0 = E[X / (a + b - X)] is the exact value stored in the gram (j0_upper); every other pair and reference uses
the second-order value f(mu) + f''(mu) Var(X) / 2. The excess overlap U is the mean over pairs of J - J0
(U_uniform, U_sign, U_magnitude), with a delete-one-category jackknife interval.

Backbone and specific weights: a kept weight is backbone if at least 90% of the other circuits keep it
(t' >= 0.9 (K - 1), t' its count among the other circuits) and specific if at most 1% do. Each circuit's backbone and
specific shares are compared with their expectation when every other circuit j keeps each weight of the layer with
probability n_j / N (uniform), within each sign, or within the weight's stratum (n_j,s / N_s); the tails are exact
Poisson-binomial sums.

Extraction noise, on the shared categories (P against P-redraw, where a repeat extraction exists): the overlap of a
category's two circuits J(P_c, R_c); the excess overlap between categories within P, within P-redraw and across the
two; and the correlation over category pairs of the excess overlap from P with that from P-redraw (the test-retest
reliability of the pair-overlap matrix).

Scope: values over all pairs or circuits present are exact (D); the intervals (delete-one-category jackknife for pair
statistics, category bootstrap for per-circuit values) are for categories beyond these (G).

Usage: python overlap_nulls.py --layer conv3 --gram-dir DIR --checkpoint PTH --out NEW_DIR
       python overlap_nulls.py --layer conv3 --gram-dir DIR --checkpoint PTH --out NEW_DIR --results RESULTS
           --arm-p FOLDER --arm-r FOLDER --taxonomy CSV
       python overlap_nulls.py --selftest
The first form takes the circuits' folders and the category tree from arms_config.json.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import category_resampling as PS
import pair_stats as PST
from arms import load_config, arm_record, input_path, retest_pair, declared_cohort, validate_cohort, add_config_arguments, InputNotAvailable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_dir, publish_file, validate_paths

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
CROSSING_GRAINS = ("taps", "unit-kernel", "unit-fanin")

LAYERS = {"conv1": "features.stage1.conv", "conv2": "features.stage2.conv", "conv3": "features.stage3.conv",
          "conv4": "features.stage4.conv", "conv5": "features.stage5.conv", "fc6": "readout.hidden1",
          "fc7": "readout.hidden2"}


# ----------------------------------------------------------------------------------------------- references

def expected_intersection(Na: np.ndarray, Nb: np.ndarray, sizes: np.ndarray):
    """Mean and variance of X = sum_s Hypergeometric(N_s; a_s, b_s) for every row pair: Na (A x S), Nb (B x S),
    sizes (S,). Returns (mu, var), each A x B."""
    Ns = sizes.astype(np.float64)
    ok = Ns > 0
    Na, Nb, Ns = Na[:, ok].astype(np.float64), Nb[:, ok].astype(np.float64), Ns[ok]
    mu = (Na / Ns) @ Nb.T
    # Var of a hypergeometric: a b (N - a)(N - b) / (N^2 (N - 1)); N = 1 gives 0
    denom = Ns ** 2 * np.maximum(Ns - 1, 1)
    var = ((Na * (Ns - Na)) / denom) @ (Nb * (Ns - Nb)).T
    return mu, var


def j0_second_order(mu, var, S):
    """E[X / (S - X)] to second order; S = a + b."""
    d = S - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(d > 0, mu / d, np.nan)
        f2 = np.where(d > 0, 2.0 * S / d ** 3, 0.0)
    return f + 0.5 * f2 * var


def pb_tails_leave_one_out(p: np.ndarray, hi: int, lo: int):
    """For each k: P(sum_{j != k} B_j >= hi) and P(sum_{j != k} B_j <= lo), B_j ~ Bernoulli(p_j). Exact dynamic
    programming over all k at once (K x (K + 1) matrix; row k skips j = k)."""
    p = np.asarray(p, dtype=np.float64)
    K = p.size
    pm = np.zeros((K, K + 1))
    pm[:, 0] = 1.0
    for j in range(K):
        q = np.full(K, p[j])
        q[j] = 0.0
        shifted = np.zeros_like(pm)
        shifted[:, 1:] = pm[:, :-1]
        pm = pm * (1.0 - q)[:, None] + shifted * q[:, None]
    hi_t = pm[:, hi:].sum(1) if hi <= K else np.zeros(K)
    lo_t = pm[:, : lo + 1].sum(1) if lo >= 0 else np.zeros(K)
    return hi_t, lo_t


def expected_class_shares(Nks: np.ndarray, sizes: np.ndarray, hi: int, lo: int):
    """Per circuit: expected backbone and specific shares of its kept routes when every other circuit j keeps each
    route of stratum s independently with probability Nks[j, s] / sizes[s]."""
    K, S = Nks.shape
    n = Nks.sum(1).astype(np.float64)
    bb, sp = np.zeros(K), np.zeros(K)
    for s in range(S):
        if sizes[s] == 0 or Nks[:, s].sum() == 0:
            continue
        t_hi, t_lo = pb_tails_leave_one_out(Nks[:, s] / float(sizes[s]), hi, lo)
        bb += Nks[:, s] * t_hi
        sp += Nks[:, s] * t_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        return bb / n, sp / n


def dyadic_block_jackknife(E, blocks=None, mask=None, unit="category"):
    """Pair mean and delete-one-unit t interval, using endpoint sufficient sums."""
    from scipy.stats import t
    if unit not in PS.UNITS:
        raise ValueError("unknown resampling unit")
    E = np.asarray(E, dtype=float)
    n = len(E)
    if E.shape != (n, n):
        raise ValueError("square category matrix required")
    labels = np.arange(n) if unit == "category" else np.asarray(blocks)
    if labels.shape != (n,):
        raise ValueError("one group label per category is required")
    groups, inv = np.unique(labels, return_inverse=True)
    g = len(groups)
    if g < 2:
        raise ValueError("at least two deletion units are required")
    i, j = np.triu_indices(n, 1)
    keep = np.isfinite(E[i, j])
    if mask is not None:
        keep &= np.asarray(mask, bool)[i, j]
    i, j, values = i[keep], j[keep], E[i[keep], j[keep]]
    if not len(values):
        raise ValueError("dyadic jackknife undefined: no eligible pairs")
    cross = inv[i] != inv[j]
    touch_sum = np.bincount(inv[i], weights=values, minlength=g) + np.bincount(inv[j[cross]], weights=values[cross], minlength=g)
    touch_count = np.bincount(inv[i], minlength=g) + np.bincount(inv[j[cross]], minlength=g)
    left = len(values) - touch_count
    if np.any(left <= 0):
        raise ValueError("dyadic jackknife undefined: a unit deletion leaves no eligible pairs")
    replicates = (values.sum() - touch_sum) / left
    mean = float(values.mean())
    se = float(np.sqrt((g-1)/g * np.square(replicates-replicates.mean()).sum()))
    half = float(t.ppf(.975, g-1) * se)
    return mean, se, mean-half, mean+half, g


def overlap_statistics(j, uniform, magnitude, blocks, positive=None, negative=None, domains=None,
                       unit="category", network="alexnet", n_resamples=10000, seed=0):
    """Further rows of the layer's pair statistics: the share of the excess over the uniform reference that the
    magnitude reference explains; the excess of positive minus negative weights (computed within each sign, Fig. 4D)
    with a jackknife interval and a two-sided t test on the jackknife standard error; and the excess within minus
    between domains (animate, inanimate) with a jackknife interval and a label-permutation p value."""
    eu, em = j - uniform, j - magnitude
    labels = np.arange(len(blocks)) if unit == "category" else np.asarray(blocks)
    rows = []
    def add(quantity, result, family="descriptive", p=None, method=None):
        rows.append({"quantity": quantity, "pairs": "all", "network": network,
            **PST.result_row(result["estimate"], ci_low=result["ci_low"], ci_high=result["ci_high"], p=p,
                family=family if family == "descriptive" else network + ":" + family,
                unit=unit, effect_unit="fraction", scope="G", n_categories=len(blocks),
                n_units=len(np.unique(labels)), n_blocks=len(np.unique(blocks)),
                method=method or "delete-one-%s jackknife; t interval" % unit)})
    def explained(u, m):
        baseline = PST.pair_mean(u)
        return 1 - PST.pair_mean(m) / baseline if baseline else np.nan
    try:
        inferred = PS.joint_block_jackknife([eu, em], labels, explained, unit=unit)
        add("magnitude_explained_share", inferred)
    except ValueError as error:
        rows.append({"quantity": "magnitude_explained_share", "pairs": "all", "network": network,
            **PST.result_row(None, family="descriptive", unit=unit, effect_unit="fraction", scope="D",
                n_categories=len(blocks), n_units=len(np.unique(labels)), n_blocks=len(np.unique(blocks)),
                method="not formed: " + str(error))})
    if positive is not None and negative is not None:
        from scipy.stats import t
        inferred = PS.block_jackknife(positive-negative, labels, unit=unit)
        se, effect = inferred["se"], inferred["estimate"]
        p = float(2*t.sf(abs(effect/se), len(np.unique(labels))-1)) if se > 0 else (1. if effect == 0 else 0.)
        add("U_positive_minus_negative", inferred, "sharing: sign contrast across 7 layers", p,
            "delete-one-%s jackknife; t interval and two-sided Wald test" % unit)
    if domains is not None:
        def contrast(current, matrix=em):
            vals = PST.pair_values(matrix)
            i, k = np.triu_indices(len(current), 1)
            within = current[i] == current[k]
            good = np.isfinite(vals)
            return float(vals[good & within].mean() - vals[good & ~within].mean()) if (good & within).any() and (good & ~within).any() else np.nan
        try:
            inferred = PST.indexed_jackknife(lambda idx: contrast(np.asarray(domains)[idx], em[np.ix_(idx, idx)]), labels, unit=unit)
            if not np.isfinite(inferred.get("ci_low", np.nan)):
                raise ValueError(inferred.get("reason", "undefined domain deletion"))
        except ValueError as error:
            inferred = dict(estimate=contrast(np.asarray(domains)), ci_low=None, ci_high=None)
            interval_status = "not formed: " + str(error)
        else:
            interval_status = "formed"
        try:
            permutation = PS.label_permutation(domains, labels, contrast, n_resamples=n_resamples, seed=seed, unit=unit)
            p, status = permutation["p"], "formed"
        except ValueError as error:
            p, status = None, "not formed: " + str(error)
        add("U_within_minus_between_domain", inferred, "sharing: domain contrast across layers", p,
            "delete-one-%s jackknife; %s domain-label permutation" % (unit, unit))
        rows[-1]["test_status"] = status
        rows[-1]["interval_status"] = interval_status
        rows[-1]["scope"] = "G" if inferred["ci_low"] is not None else "D"
    return rows


def sign_excess(packed, stratum_labels, strata_sizes, counts, k):
    """The excess overlap of each pair within each sign, positive then negative weights: Jaccard overlap of the kept
    weights of that sign minus its second-order expectation when each circuit's kept weights of that sign are placed
    at random within that sign's weight-size strata."""
    from mask_gram import PackedRuns, gram
    sign = stratum_labels >= len(strata_sizes)//2
    output = []
    for negative in (False, True):
        positions = sign == negative
        width = int(positions.sum())
        bits = np.stack([np.packbits(np.unpackbits(row, count=len(sign))[positions]) for row in packed[:k]])
        intersections = gram(PackedRuns(bits, width)).astype(float)
        size = np.diag(intersections)
        sums = size[:, None] + size[None, :]
        j = np.divide(intersections, sums-intersections,
                      out=np.full_like(intersections, np.nan), where=sums > intersections)
        choose = (np.arange(len(strata_sizes)) >= len(strata_sizes)//2) == negative
        mu, var = expected_intersection(counts[:k, choose], counts[:k, choose], strata_sizes[choose])
        output.append(j-j0_second_order(mu, var, sums))
    return output


def related_crossing_statistics(matrix, blocks, related_labels, grain="taps", unit="category",
                                network="alexnet", n_resamples=10000, seed=0):
    """The mean of a category-pair crossing matrix over related pairs (same label) minus unrelated pairs, with a
    delete-one-category jackknife interval and a label-permutation p value."""
    labels = np.asarray(related_labels)
    units = np.arange(len(labels)) if unit == "category" else np.asarray(blocks)
    def contrast(values, current):
        i, j = np.triu_indices(len(current), 1)
        v = values[i, j]
        related, good = current[i] == current[j], np.isfinite(v)
        if not (good & related).any() or not (good & ~related).any():
            return np.nan
        return float(v[good & related].mean()-v[good & ~related].mean())
    try:
        jk = PST.indexed_jackknife(lambda idx: contrast(matrix[np.ix_(idx, idx)], labels[idx]), units, unit=unit)
        if not np.isfinite(jk.get("ci_low", np.nan)):
            raise ValueError(jk.get("reason", "undefined related-label deletion"))
    except ValueError as error:
        jk = dict(ci_low=None, ci_high=None)
        interval_status = "not formed: " + str(error)
    else:
        interval_status = "formed"
    try:
        perm = PS.label_permutation(labels, units, lambda current: contrast(matrix, current),
                                    n_resamples=n_resamples, seed=seed, unit=unit)
        p, status = perm["p"], "formed"
    except ValueError as error:
        p, status = None, "not formed: " + str(error)
    return {"quantity": "crossing_related_minus_unrelated", "pairs": "all", "grain": grain,
        "network": network, "test_status": status, "interval_status": interval_status,
        **PST.result_row(contrast(matrix, labels), ci_low=jk["ci_low"], ci_high=jk["ci_high"], p=p,
            family=network + ":sharing: related crossing across layers x grains", unit=unit, effect_unit="fraction",
            scope="G" if jk["ci_low"] is not None and np.isfinite(jk["ci_low"]) else "D", n_categories=len(blocks), n_units=len(np.unique(units)),
            n_blocks=len(np.unique(blocks)), method="delete-one-%s jackknife; %s label permutation" % (unit, unit))}


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() > 2 else np.nan


# ----------------------------------------------------------------------------------------------- one layer

def strata_labels(w: np.ndarray):
    """The stratum of each weight (sign x |w| decile within the sign), the number of strata and their definition."""
    from strata import route_strata
    st = route_strata(w)
    return np.asarray(st.labels, dtype=np.int64), int(st.n_strata), st.definition


def verify_gram_masks(z, hashes, packed, K):
    """Check that the masks read are those the gram was computed from: the P masks' sha256 in order, and the repeat
    extractions' sha256 or, where the gram has none, their intersection counts."""
    hashes = np.asarray(hashes, dtype=str)
    if hashes.shape != (len(packed),) or K > len(packed):
        raise ValueError("loaded mask/hash lengths differ")
    recorded = np.asarray(z["mask_sha256"]).astype(str)
    if recorded.shape != (K,) or not np.array_equal(recorded, hashes[:K]):
        raise ValueError("P masks differ from the Gram's complete ordered mask_sha256 array")
    Kr = len(packed) - K
    if "redraw_mask_sha256" in z.files:
        recorded_redraw = np.asarray(z["redraw_mask_sha256"]).astype(str)
        if recorded_redraw.shape != (Kr,) or not np.array_equal(recorded_redraw, hashes[K:]):
            raise ValueError("redraw masks differ from the Gram's complete ordered redraw_mask_sha256 array")
    elif Kr:
        intersections = z["I_all"]
        if intersections.shape != (K + Kr, K + Kr):
            raise ValueError("Gram I_all dimensions differ from the loaded masks")
        popcount = np.array([bin(int(i)).count("1") for i in range(256)], dtype=np.uint8)
        for i in range(K, K + Kr):
            row = np.fromiter((int(popcount[np.bitwise_and(packed[i], other)].sum(dtype=np.int64))
                               for other in packed), dtype=np.int64, count=K + Kr)
            if not np.array_equal(row, intersections[i]):
                raise ValueError(f"redraw row {i - K}: loaded-mask intersections differ from Gram I_all")


def run(a) -> int:
    check_outputs(dirs=[a.out], inputs=[a.results, a.taxonomy, a.checkpoint, a.gram_dir,
        getattr(a, "gram_file", None), getattr(a, "arms_config", None)])
    import torch
    if not getattr(a, "arm_p", None) or not getattr(a, "arm_r", None):
        conf = load_config(getattr(a, "arms_config", None))
        price = getattr(a, "price", "v66")
        pr = arm_record(conf, a.network, "P", price=price)
        try:
            pair = retest_pair(conf, a.network, price)
            first, second = pair["primary"], pair["retest"]
            pr = arm_record(conf, a.network, first["arm"], price=price, run=first["run"])
            rr = arm_record(conf, a.network, second["arm"], price=price, run=second["run"])
        except InputNotAvailable:
            rr = None
        a.taxonomy = a.taxonomy or str(input_path(conf, "taxonomy"))
        a.results = a.results or str(pr["results_root"])
        a.arm_p = a.arm_p or pr["arm_folder"]
        a.arm_r = a.arm_r or (rr["arm_folder"] if rr else None)
        redraw_results = str(rr["results_root"]) if rr else a.results
    else:
        redraw_results = a.results
    import circuit_masks as RT
    out = Path(a.out)
    check_outputs(dirs=[out], inputs=[a.results, redraw_results, a.taxonomy, a.checkpoint,
        a.gram_file, a.gram_dir])
    gram_path = Path(a.gram_file) if a.gram_file else Path(a.gram_dir) / f"gram_{a.layer}.npz"
    if a.network == "alexnet" and a.layer:
        lid = LAYERS[a.layer]
    else:
        lid = json.loads(gram_path.with_suffix(".json").read_text())["layer_id"]
        a.layer = a.layer or str(np.load(gram_path)["layer"])
    z = np.load(gram_path)
    cats, j0u = z["categories"], z["j0_upper"]
    if "redraw_categories" in z.files and z["redraw_categories"].size and "I_all" not in z.files:
        raise ValueError("redraw categories declared but I_all missing")
    if "I_all" in z.files and "redraw_categories" in z.files and z["redraw_categories"].size:
        rcats, I_all = z["redraw_categories"], z["I_all"].astype(np.float64)
        if not (np.all(np.diff(cats) > 0) and np.all(np.diff(rcats) > 0) and np.isin(rcats, cats).all()):
            raise ValueError("categories must be sorted, unique, redraws a subset of P")
    else:
        rcats, I_all = np.zeros(0, np.int64), z["I"].astype(np.float64)
    K, Kr = cats.size, rcats.size
    partial_metadata = {"partial": False}
    if "conf" in locals():
        expected = declared_cohort(conf, a.network, "P", price=price, record=pr)
        partial_metadata = validate_cohort(cats, expected, getattr(a, "allow_partial", False))
        if Kr and rr:
            expected_retest = declared_cohort(conf, a.network, record={"cohort": pair["cohort"]})
            other_metadata = validate_cohort(rcats, expected_retest, getattr(a, "allow_partial", False))
            partial_metadata["partial"] = partial_metadata["partial"] or other_metadata["partial"]
    if Kr and not a.arm_r:
        raise ValueError("Gram declares second-run categories but no matching second-run arm is configured")
    PSET = "P_1000" if a.network == "alexnet" else f"P_all_{a.network}"
    assert I_all.shape == (K + Kr, K + Kr), I_all.shape
    layout = RT.tree_layout(a.network)
    spec = next(s for s in layout.layers if s.layer_id == lid)
    if not a.checkpoint:
        raise ValueError("--checkpoint must name the checkpoint the circuits were extracted from")
    ck = a.checkpoint
    validate_paths([out], [gram_path, a.results, a.taxonomy, ck])
    import hashlib as _h
    ck_sha = _h.sha256(Path(ck).read_bytes()).hexdigest()       # the checkpoint the circuits were extracted from, only
    if not ck_sha.startswith(RT.CHECKPOINT_FILE[a.network].split("-")[-1].split(".")[0]):
        raise ValueError(f"{ck}: sha256 {ck_sha[:8]} is not the {a.network} checkpoint the circuits were extracted from")
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    w = sd[spec.module_name + ".weight"].detach().cpu().numpy()
    assert tuple(w.shape) == spec.shape
    lab, S, sdef = strata_labels(w.reshape(-1))
    N = spec.size
    sizes = np.bincount(lab, minlength=S)
    sign_of_stratum = (np.arange(S) >= S // 2).astype(int)        # label = polarity * 10 + decile
    lab_sign = sign_of_stratum[lab]
    res = Path(a.results)
    rows = [(c, a.arm_p, res) for c in cats] + [(c, a.arm_r, Path(redraw_results)) for c in rcats]
    Nks = np.zeros((K + Kr, S), np.int64)
    t = np.zeros(N, np.uint16)
    packed, mask_hashes = [], []
    for i, (c, arm, root) in enumerate(rows):
        d = root / f"{a.network}_tap_query{int(c)}" / arm
        rec = json.loads((d / "receipt.json").read_text())
        mask_path = d / "final_mask.packbits"
        mask_hash = _h.sha256()
        with mask_path.open("rb") as mask_file:
            for chunk in iter(lambda: mask_file.read(1024 * 1024), b""):
                mask_hash.update(chunk)
        mask_hashes.append(mask_hash.hexdigest())
        bits = RT.load_verified_mask(mask_path, rec, layout)[spec.offset:spec.offset + spec.size]
        if int(bits.sum()) != int(I_all[i, i]):
            raise ValueError(f"category {c} {arm}: popcount {int(bits.sum())} != gram diagonal {int(I_all[i, i])}")
        Nks[i] = np.bincount(lab[bits], minlength=S)
        if i < K:
            t += bits
        packed.append(np.packbits(bits))
        if i % 200 == 0:
            print(f"  read {i + 1}/{K + Kr} masks", flush=True)
    if Kr and all(mask_hashes[np.searchsorted(cats, c)] == mask_hashes[K+i] for i, c in enumerate(rcats)):
        raise ValueError("the two named retest runs have byte-identical masks for every category")
    verify_gram_masks(z, mask_hashes, packed, K)
    packed = packed[:K]
    # ---- backbone and specific weights, observed (t' = t_r - 1, the count among the other circuits, on kept weights)
    hi, lo = math.ceil(0.9 * (K - 1) - 1e-9), math.floor(0.01 * (K - 1) + 1e-9)
    is_bb = (t.astype(np.int64) - 1) >= hi
    is_sp = (t.astype(np.int64) - 1) <= lo
    obs_bb_s = np.zeros((K, S), np.int64)
    obs_sp_s = np.zeros((K, S), np.int64)
    for k in range(K):
        bits = np.unpackbits(packed[k], count=N).astype(bool)
        obs_bb_s[k] = np.bincount(lab[bits & is_bb], minlength=S)
        obs_sp_s[k] = np.bincount(lab[bits & is_sp], minlength=S)
    positive_excess, negative_excess = sign_excess(packed, lab, sizes, Nks, K)
    del packed
    n = Nks[:K].sum(1).astype(np.float64)
    if np.any(n == 0):                                         # an undefined share stops the run; it is not dropped
        raise ValueError(f"{int((n == 0).sum())} circuits keep no route in {lid}")
    obs_bb, obs_sp = obs_bb_s.sum(1) / n, obs_sp_s.sum(1) / n
    Nsign = np.stack([Nks[:, sign_of_stratum == g].sum(1) for g in (0, 1)], 1)
    sizes_sign = np.array([sizes[sign_of_stratum == g].sum() for g in (0, 1)])
    Nuni = Nks.sum(1, keepdims=True)
    sizes_uni = np.array([N])
    null_bb, null_sp = {}, {}
    for name, M_, sz in (("uniform", Nuni[:K], sizes_uni), ("sign", Nsign[:K], sizes_sign), ("magnitude", Nks[:K], sizes)):
        null_bb[name], null_sp[name] = expected_class_shares(M_, sz, hi, lo)
    # ---- pair overlap under the three references (P and P-redraw rows at once)
    a_ = np.diag(I_all)
    Ssum = a_[:, None] + a_[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(Ssum - I_all > 0, I_all / (Ssum - I_all), np.nan)
    J0 = {}
    for name, M_, sz in (("uniform", Nuni, sizes_uni), ("sign", Nsign, sizes_sign), ("magnitude", Nks, sizes)):
        mu, var = expected_intersection(M_, M_, sz)
        J0[name] = j0_second_order(mu, var, Ssum)
    iu = np.triu_indices(K, 1)
    j0_check = float(np.nanmax(np.abs(J0["uniform"][:K, :K][iu] - j0u)))
    # the P-P pairs take the gram's exact uniform J0; the other pairs and the sign and magnitude references keep the
    # second-order value
    Ju = J0["uniform"].copy(); Ju[:K, :K][iu] = j0u; Ju[:K, :K][(iu[1], iu[0])] = j0u; J0["uniform"] = Ju
    tax = {int(r["category"]): r for r in __import__("csv").DictReader(open(a.taxonomy))}
    blocks = np.array([tax[int(c)]["block50"] for c in cats])
    unit = getattr(a, "unit", "category")
    if unit == "category":
        resampling = cats
    else:
        column = "animate" if unit == "domain" and "domain" not in tax[int(cats[0])] else unit
        resampling = np.array([tax[int(c)][column] for c in cats])
    anim = np.array([int(tax[int(c)]["animate"]) == 1 for c in cats])
    same_block = blocks[:, None] == blocks[None, :]
    masks = {"all": None, "within_block": same_block, "cross_block": ~same_block,
             "within_animate": np.outer(anim, anim), "within_inanimate": np.outer(~anim, ~anim),
             "animate_vs_inanimate": np.outer(anim, ~anim) | np.outer(~anim, anim)}
    summary = []
    JP = J[:K, :K]
    extra = overlap_statistics(JP, J0["uniform"][:K, :K], J0["magnitude"][:K, :K], resampling,
        positive=positive_excess, negative=negative_excess, domains=np.array([tax[int(c)].get("domain", tax[int(c)]["animate"]) for c in cats]), unit=unit, network=a.network)
    for filename in getattr(a, "crossing_pairs", None) or []:
        crossing = np.load(filename)
        cc = crossing["categories"]
        common = np.intersect1d(cats, cc)
        ci, ti = np.searchsorted(cc, common), np.searchsorted(cats, common)
        extra.append(related_crossing_statistics(crossing["crossing"][np.ix_(ci, ci)], resampling[ti], blocks[ti], unit=unit, network=a.network,
            grain=str(crossing["grain"].item()) if "grain" in crossing.files else "taps"))
    summary.extend({"layer": a.layer, "set": PSET, "value_D": row["estimate"],
                    "lo95_G": row["ci_low"], "hi95_G": row["ci_high"], **row} for row in extra)
    for name in ("uniform", "sign", "magnitude"):
        E = JP - J0[name][:K, :K]
        for mname, mk in masks.items():
            m, se, lo95, hi95, G = dyadic_block_jackknife(E, resampling, mk, unit=unit)
            summary.append({"layer": a.layer, "set": PSET, "quantity": f"U_{name}", "pairs": mname,
                            "value_D": m, "jackknife_se_G": se, "lo95_G": lo95, "hi95_G": hi95, "n_blocks": G,
                            "n_categories": K})
    summary.append({"layer": a.layer, "set": PSET, "quantity": "mean_J", "pairs": "all",
                    "value_D": float(np.nanmean(JP[iu])), "n_categories": K})
    for name in ("uniform", "sign", "magnitude"):
        summary.append({"layer": a.layer, "set": PSET, "quantity": f"mean_J0_{name}", "pairs": "all",
                        "value_D": float(np.nanmean(J0[name][:K, :K][iu])), "n_categories": K})
    # ---- shared categories: P against P-redraw (only where the gram holds a repeat extraction)
    if Kr:
        pidx = np.searchsorted(cats, rcats)
        assert np.array_equal(cats[pidx], rcats)
        ridx = K + np.arange(Kr)
        rb = resampling[pidx]
        iu100 = np.triu_indices(Kr, 1)
        for name in ("uniform", "sign", "magnitude"):
            Eall = J - J0[name]
            EPP = Eall[np.ix_(pidx, pidx)]
            ERR = Eall[np.ix_(ridx, ridx)]
            EPR = Eall[np.ix_(pidx, ridx)]
            EPRs = 0.5 * (EPR + EPR.T)
            retest = np.diag(EPR)
            vals = {"U_P_shared100": EPP[iu100], "U_Predraw_shared100": ERR[iu100], "U_cross_extraction": EPRs[iu100],
                    "retest_excess_J(Pc,Rc)": retest}
            for q, v in vals.items():
                interval = PS.block_bootstrap(v, rb, unit=unit) if q.startswith("retest_") else PS.block_jackknife(
                    {"U_P_shared100": EPP, "U_Predraw_shared100": ERR, "U_cross_extraction": EPRs}[q], rb, unit=unit)
                summary.append({"layer": a.layer, "set": "shared100", "quantity": f"{q}_{name}", "pairs": "all",
                                "value_D": float(np.nanmean(v)), "n_categories": Kr,
                                "lo95_G": interval["ci_low"], "hi95_G": interval["ci_high"]})
            summary.append({"layer": a.layer, "set": "shared100", "quantity": f"reliability_pearson_PP_vs_RR_{name}",
                            "pairs": "all", "value_D": pearson(EPP[iu100], ERR[iu100]), "n_categories": Kr})
            summary.append({"layer": a.layer, "set": "shared100", "quantity": f"reliability_spearman_PP_vs_RR_{name}",
                            "pairs": "all", "value_D": spearman(EPP[iu100], ERR[iu100]), "n_categories": Kr})
            interval = PS.block_jackknife(EPP-ERR, rb, unit=unit)
            summary.append({"layer": a.layer, "set": "shared100", "quantity": f"U_P_minus_U_Predraw_{name}",
                            "pairs": "all", "value_D": float(np.nanmean(EPP[iu100]) - np.nanmean(ERR[iu100])),
                            "n_categories": Kr, "lo95_G": interval["ci_low"], "hi95_G": interval["ci_high"]})
            mb = blocks[pidx, None] == blocks[None, pidx]
            for mname, mk in (("within_block", mb), ("cross_block", ~mb)):
                sel = mk[iu100]
                summary.append({"layer": a.layer, "set": "shared100", "quantity": f"U_P_shared100_{name}",
                                "pairs": mname, "value_D": float(np.nanmean(EPP[iu100][sel])), "n_categories": Kr,
                                "lo95_G": dyadic_block_jackknife(EPP, rb, mk, unit=unit)[2],
                                "hi95_G": dyadic_block_jackknife(EPP, rb, mk, unit=unit)[3]})
                summary.append({"layer": a.layer, "set": "shared100", "quantity": f"U_Predraw_shared100_{name}",
                                "pairs": mname, "value_D": float(np.nanmean(ERR[iu100][sel])), "n_categories": Kr,
                                "lo95_G": dyadic_block_jackknife(ERR, rb, mk, unit=unit)[2],
                                "hi95_G": dyadic_block_jackknife(ERR, rb, mk, unit=unit)[3]})
        JPR = J[np.ix_(pidx, ridx)]
        for q, v in (("retest_J(Pc,Rc)", np.diag(JPR)), ("between_J_P", J[np.ix_(pidx, pidx)][iu100]),
                     ("between_J_Predraw", J[np.ix_(ridx, ridx)][iu100])):
            summary.append({"layer": a.layer, "set": "shared100", "quantity": q, "pairs": "all",
                            "value_D": float(np.nanmean(v)), "n_categories": Kr,
                            "min": float(np.nanmin(v)), "max": float(np.nanmax(v))})
        summary.append({"layer": a.layer, "set": "shared100", "quantity": "pearson_J_PP_vs_RR", "pairs": "all",
                        "value_D": pearson(J[np.ix_(pidx, pidx)][iu100], J[np.ix_(ridx, ridx)][iu100]),
                        "n_categories": Kr})
    # ---- backbone summaries
    for cls, obs, nul in (("backbone", obs_bb, null_bb), ("specific", obs_sp, null_sp)):
        bs = PS.block_bootstrap(obs, resampling, unit=unit)
        m, lo95, hi95 = bs["estimate"], bs["ci_low"], bs["ci_high"]
        summary.append({"layer": a.layer, "set": PSET, "quantity": f"{cls}_share_observed", "pairs": "circuits",
                        "value_D": m, "lo95_G": lo95, "hi95_G": hi95, "n_categories": K})
        for name in ("uniform", "sign", "magnitude"):
            ex = obs - nul[name]
            bs = PS.block_bootstrap(ex, resampling, unit=unit)
            m, lo95, hi95 = bs["estimate"], bs["ci_low"], bs["ci_high"]
            summary.append({"layer": a.layer, "set": PSET, "quantity": f"{cls}_share_null_{name}",
                            "pairs": "circuits", "value_D": float(np.nanmean(nul[name])), "n_categories": K})
            summary.append({"layer": a.layer, "set": PSET, "quantity": f"{cls}_excess_over_{name}",
                            "pairs": "circuits", "value_D": m, "lo95_G": lo95, "hi95_G": hi95, "n_categories": K})
    # backbone share of the kept weights in each stratum (pooled over circuits)
    by_stratum = []
    for s in range(S):
        kept = int(Nks[:K, s].sum())
        by_stratum.append({"layer": a.layer, "stratum": s, "polarity": "negative" if s >= S // 2 else "positive",
                           "scope": "D", "n_categories": K, "n_blocks": len(np.unique(blocks)),
                           "abs_w_decile": s % (S // 2), "routes": int(sizes[s]),
                           "kept_share_mean": float((Nks[:K, s] / max(sizes[s], 1)).mean()),
                           "backbone_share_of_kept": float(obs_bb_s[:, s].sum() / kept) if kept else np.nan,
                           "specific_share_of_kept": float(obs_sp_s[:, s].sum() / kept) if kept else np.nan})
    import pandas as pd
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out.name + ".partial-", dir=out.parent))
    for row in summary:
        if "family" in row:
            continue
        category_level = row["pairs"] == "circuits" or row["quantity"].startswith(("retest_J(", "retest_excess_"))
        row.update(PST.result_row(row.get("value_D"), ci_low=row.get("lo95_G"), ci_high=row.get("hi95_G"),
            family="descriptive", unit=unit, effect_unit="fraction", scope="G" if "lo95_G" in row else "D",
            n_categories=row["n_categories"], n_units=len(np.unique(resampling)) if row["set"] == PSET else len(np.unique(rb)),
            n_blocks=len(np.unique(blocks)) if row["set"] == PSET else len(np.unique(blocks[pidx])),
            method="percentile %s bootstrap (10000)" % unit if category_level else "delete-one-%s jackknife; t interval" % unit))
    for row in summary:
        row.update(partial_metadata)
    publish_file(pd.DataFrame(summary), tmp / "summary.csv")
    for row in by_stratum:
        row.update(partial_metadata, unit=unit, n_units=len(np.unique(resampling)), method="exact stratum summary")
    publish_file(pd.DataFrame(by_stratum), tmp / "by_stratum.csv")
    publish_file(pd.DataFrame({"category": cats, "block50": blocks, "scope": "D", "n_categories": K,
                  "n_blocks": len(np.unique(blocks)), "unit": unit, "n_units": len(np.unique(resampling)),
                  **partial_metadata, "method": "exact circuit summary", "kept": n.astype(np.int64), "backbone_obs": obs_bb,
                  "specific_obs": obs_sp, **{f"backbone_null_{k}": v for k, v in null_bb.items()},
                  **{f"specific_null_{k}": v for k, v in null_sp.items()}}), tmp / "per_circuit.csv")
    pair_bytes = io.BytesIO()
    np.savez_compressed(pair_bytes, categories=cats, redraw_categories=rcats,
                        J0_sign=J0["sign"].astype(np.float32), J0_magnitude=J0["magnitude"].astype(np.float32),
                        J0_uniform=J0["uniform"].astype(np.float32), Nks=Nks, strata_sizes=sizes)
    publish_file(pair_bytes.getvalue(), tmp / "pairs.npz")
    meta = {"network": a.network, "layer": a.layer, "layer_id": lid, "n_routes": int(N), "K": int(K), "K_redraw": int(Kr),
            "strata": sdef, "n_strata": S, "strata_sizes": sizes.tolist(),
            "thresholds": {"backbone_t_other_ge": hi, "specific_t_other_le": lo},
            "j0_uniform_vs_exact_max_abs_diff": j0_check, "gram": str(gram_path),
            "results": str(res), "arm_p": a.arm_p, "arm_r": a.arm_r, "checkpoint": str(ck), "checkpoint_sha256": ck_sha,
            "taxonomy": str(a.taxonomy),
            "expected_crossing_grains": list(getattr(a, "crossing_grains", CROSSING_GRAINS)),
            "expected_layers": [RT.short_name(a.network, s.layer_id) for s in layout.layers
                                if getattr(s, "kind", None) != "residual_stream_gate"],
            "scope": "D: exact over all pairs/circuits present; G: delete-one-unit jackknife (pairs) or unit "
                     "bootstrap (circuits), for categories beyond these; extraction noise: shared-100 P vs P-redraw",
            "unit": unit, "script": str(Path(__file__).resolve())}
    publish_file(json.dumps(meta, indent=1).encode(), tmp / "meta.json")
    publish_dir(tmp, out)
    print(json.dumps({k: meta[k] for k in ("layer", "K", "j0_uniform_vs_exact_max_abs_diff")}))
    return 0


# ----------------------------------------------------------------------------------------------- checks on planted data

def selftest() -> int:
    rng = np.random.default_rng(1)
    # (1) the second-order J0 against a simulation of the stratified (magnitude) reference
    sizes = np.array([400, 300, 200, 100])
    na, nb = np.array([[40, 150, 20, 90]]), np.array([[200, 30, 150, 50]])
    mu, var = expected_intersection(na, nb, sizes)
    S = na.sum() + nb.sum()
    j0 = j0_second_order(mu, var, np.array([[S]]))[0, 0]
    sims = []
    for _ in range(20000):
        x = sum(rng.hypergeometric(nb[0, s], sizes[s] - nb[0, s], na[0, s]) for s in range(4))
        sims.append(x / (S - x))
    sim = float(np.mean(sims))
    assert abs(j0 - sim) < 3 * np.std(sims) / math.sqrt(len(sims)) + 2e-4, (j0, sim)
    # (2) uniform second-order J0 against the exact windowed hypergeometric expectation
    try:
        sys.path.insert(0, str(HERE.parent / "animacy"))
        from library_statistics import exact_J0
        for (aa, bb, NN) in ((50, 70, 1000), (900, 950, 1000), (3, 400, 23232), (20000, 30000, 600000)):
            m2, v2 = expected_intersection(np.array([[aa]]), np.array([[bb]]), np.array([NN]))
            got = j0_second_order(m2, v2, np.array([[aa + bb]]))[0, 0]
            assert abs(got - exact_J0(aa, bb, NN)) < 5e-4 * max(1, got), (aa, bb, NN, got, exact_J0(aa, bb, NN))
        print("selftest: uniform J0 matches exact_J0")
    except ImportError:
        print("selftest: analysis/animacy/library_statistics.py not importable; exact_J0 comparison skipped")
    # (3) planted circuits that keep weights independently, at rates that depend only on the stratum
    #     -> excess over the uniform reference > 0, over the magnitude reference ~ 0
    Nr, K = 6000, 60
    lab = rng.integers(0, 4, Nr)
    sizes = np.bincount(lab, minlength=4)
    rate = np.array([0.05, 0.15, 0.4, 0.8])
    X = rng.random((K, Nr)) < rate[lab][None, :] * rng.uniform(0.7, 1.3, (K, 1))
    I = X.astype(float) @ X.T.astype(float)
    a_ = np.diag(I)
    Ss = a_[:, None] + a_[None, :]
    J = I / (Ss - I)
    Nks = np.stack([np.bincount(lab[x], minlength=4) for x in X])
    iu = np.triu_indices(K, 1)
    m_u, v_u = expected_intersection(Nks.sum(1, keepdims=True), Nks.sum(1, keepdims=True), np.array([Nr]))
    m_m, v_m = expected_intersection(Nks, Nks, sizes)
    Uu = np.mean((J - j0_second_order(m_u, v_u, Ss))[iu])
    Um = np.mean((J - j0_second_order(m_m, v_m, Ss))[iu])
    assert Uu > 0.05 and abs(Um) < 0.005, (Uu, Um)
    print(f"selftest: planted magnitude library U_uniform {Uu:.4f}, U_magnitude {Um:.5f}")
    # (4) planted category-specific shared weights remain in excess of the magnitude reference
    X2 = X.copy()
    X2[: K // 2, :300] = True                                                  # half the circuits share 300 weights
    I2 = X2.astype(float) @ X2.T.astype(float)
    a2 = np.diag(I2)
    S2 = a2[:, None] + a2[None, :]
    N2 = np.stack([np.bincount(lab[x], minlength=4) for x in X2])
    m2, v2 = expected_intersection(N2, N2, sizes)
    E2 = I2 / (S2 - I2) - j0_second_order(m2, v2, S2)
    assert np.mean(E2[: K // 2, : K // 2][np.triu_indices(K // 2, 1)]) > 0.02
    # (5) Poisson-binomial leave-one-out tails against Monte Carlo
    p = rng.uniform(0.6, 0.99, 40)
    hi_t, lo_t = pb_tails_leave_one_out(p, 36, 25)
    draws = rng.random((200000, 40)) < p
    tot = draws.sum(1)
    for k in (0, 7, 39):
        tk = tot - draws[:, k]
        assert abs(hi_t[k] - (tk >= 36).mean()) < 0.005 and abs(lo_t[k] - (tk <= 25).mean()) < 0.005
    # (6) the dyadic block jackknife's point value is the plain pair mean
    E = rng.normal(size=(30, 30))
    E = (E + E.T) / 2
    m, se, _, _, G = dyadic_block_jackknife(E, np.repeat(np.arange(6), 5), unit="block50")
    assert abs(m - E[np.triu_indices(30, 1)].mean()) < 1e-12 and se > 0 and G == 6
    print("selftest: all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the checks on planted data and exit")
    ap.add_argument("--layer", help="AlexNet layer (conv1..fc7), read from --gram-dir; omit with --gram-file")
    ap.add_argument("--gram-dir", help="folder holding gram_<layer>.npz")
    ap.add_argument("--gram-file", help="one gram_<layer>.npz, beside a .json that names its layer; any network")
    ap.add_argument("--network", default="alexnet", choices=("alexnet", "vgg11", "resnet18", "resnet50v1", "resnet50v2"))
    ap.add_argument("--results", help="the circuits' results folder (default: from arms_config.json)")
    ap.add_argument("--arm-p", help="the P circuits' folder name (default: from arms_config.json)")
    ap.add_argument("--arm-r", help="the repeat extractions' folder name (default: from arms_config.json)")
    add_config_arguments(ap)
    ap.add_argument("--checkpoint", help="the torchvision checkpoint the circuits were extracted from")
    ap.add_argument("--taxonomy", help="the category tree table (default: from arms_config.json)")
    ap.add_argument("--out", help="the new output folder")
    ap.add_argument("--crossing-grains", nargs="+", default=list(CROSSING_GRAINS),
                    help="the grains of the crossing matrices expected over all layers (recorded in meta.json)")
    ap.add_argument("--crossing-pairs", action="append",
                    help="an NPZ (categories, crossing, grain) holding a category-pair crossing matrix; adds its "
                         "related-minus-unrelated contrast (related: same subgroup)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
