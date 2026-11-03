"""Fig. 6E: the agreement of AlexNet's wiring with each other network's, and its ceiling from repeat extractions.

For AlexNet's conv5, fc6 and fc7 and every layer of another network: the Pearson correlation over category pairs of
the two wirings' excess overlap (over the sign- and weight-size-matched reference), without and with node adjustment,
on all the categories both have. On the shared categories, where repeat extractions are given: each network's
reliability (the correlation of its wiring with its repeat extraction's), the ceiling sqrt(r_A r_X), the correlation
divided by the ceiling, and the correlation minus AlexNet's reliability. Intervals: joint delete-one-category
jackknife, refitting the node adjustment in every deletion. The repeat extractions are named in --retest-manifest and
checked against arms_config.json's retest pairs and, where the masks are present, against the masks.

Usage: cross_network_agreement.py --alex-grams DIR --alex-pairs DIR --other DIR --taxonomy CSV --out CSV
           [--alex-retest-grams DIR --alex-retest-pairs DIR] [--other-retest DIR] [--retest-manifest JSON]
--other holds <network>__gram_<layer>.npz and <network>__pairs_<layer>.npz.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_share as TS  # noqa: E402
import pair_stats as PST  # noqa: E402
import category_resampling as PS  # noqa: E402
import arms as AM  # noqa: E402
from tree_infer import backfit_adjust, category_units  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_file, validate_paths, check_outputs  # noqa: E402


def excess(g, p):
    """A layer's categories and its matrix of excess overlap J - J0 (J0 the magnitude reference), diagonal NaN."""
    z, pz = np.load(g), np.load(p)
    cats = z["categories"]; K = cats.size
    assert np.array_equal(cats, pz["categories"])
    I = (z["I_all"] if "I_all" in z.files else z["I"]).astype(np.float64)[:K, :K]
    d = np.diag(I); S = d[:, None] + d[None, :]
    J = np.where(S - I > 0, I / np.where(S - I > 0, S - I, 1), np.nan)
    E = J - pz["J0_magnitude"].astype(np.float64)[:K, :K]
    np.fill_diagonal(E, np.nan)
    return cats, E


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return np.nan, np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), float(spearmanr(a[ok], b[ok]).correlation)


def fit_matrix(matrix):
    """The node-adjusted matrix (tree_infer.backfit_adjust over the finite pairs); an error if the fit fails."""
    adjusted, formed = backfit_adjust(matrix, np.isfinite(matrix) & ~np.eye(len(matrix), dtype=bool), return_convergence=True)
    if not formed:
        raise ValueError("node adjustment not formed")
    return adjusted


def verify_retest_manifest(manifest, network, primary, retest, primary_pairs=None, retest_pairs=None):
    """Check the manifest's entry for a network: both runs named, at the same price, in distinct folders that are
    the folders given. Runs are told apart by their names, not by the receipts' random_seed, which can be zero for
    both."""
    if not manifest:
        return False
    try:
        record = manifest["networks"][network]
        if any(record.get(k) in (None, "") for k in ("arm_primary", "arm_retest", "price_primary", "price_retest")):
            raise ValueError("declared retest run names and price metadata must be present")
        if record["price_primary"] != record["price_retest"]:
            raise ValueError("test-retest runs must use the same price")
        for key, actual in (("primary_directory", primary), ("retest_directory", retest)):
            if Path(record[key]).resolve() != Path(actual).resolve():
                raise ValueError("test-retest manifest directory differs from the folder given")
        if Path(primary).resolve() == Path(retest).resolve():
            raise ValueError("test-retest runs must have distinct declared directories")
        if primary_pairs is not None:
            for key, actual in (("primary_pairs_directory", primary_pairs), ("retest_pairs_directory", retest_pairs)):
                if Path(record[key]).resolve() != Path(actual).resolve():
                    raise ValueError("test-retest manifest pair directory differs from the folder given")
    except KeyError as error:
        raise ValueError("test-retest manifest is missing %s" % error)
    return True


def analyze_pair_matrices(x, y, blocks=None, x_retest=None, y_retest=None,
                          node_adjusted=False, unit="category"):
    """The raw correlation and, with repeat extractions, the reliabilities, ceiling, corrected correlation and the
    difference from AlexNet's reliability, each with a joint delete-one-category jackknife interval. The raw
    correlation uses the pairs finite in its two matrices; the others use the pairs finite in all four."""
    matrices = [np.asarray(x, float), np.asarray(y, float), x_retest, y_retest]
    def fitted(m):
        return fit_matrix(m) if node_adjusted else m
    def metric(kind, *reduced):
        values = {i: fitted(m) for i, m in zip(present, reduced)}
        if kind == "raw":
            return PST.correlation(values[0], values[1])
        common = np.logical_and.reduce([np.isfinite(m) for m in values.values()])
        values = {i: np.where(common, m, np.nan) for i, m in values.items()}
        raw = PST.correlation(values[0], values[1])
        ra = PST.correlation(values[0], values[2]) if 2 in values else np.nan
        rb = PST.correlation(values[1], values[3]) if 3 in values else np.nan
        if kind == "rel_a":
            return ra
        if kind == "rel_b":
            return rb
        if kind == "ceiling":
            return np.sqrt(ra*rb) if ra > 0 and rb > 0 else np.nan
        if kind == "difference":
            return raw-ra
        return raw / np.sqrt(ra*rb) if ra > 0 and rb > 0 else np.nan
    # The raw correlation keeps its own finite pairs, not those it shares with the repeat extractions.
    present = [0, 1]
    raw = PS.joint_block_jackknife(matrices[:2], blocks, lambda *m: metric("raw", *m), unit=unit)
    out = dict(estimate=raw["estimate"], ci_low=raw["ci_low"], ci_high=raw["ci_high"], raw_r=raw["estimate"],
        correction="not formed", correction_reason="one or both networks lack a declared same-price second run",
        reliability_alexnet=None, reliability_other=None, corrected_r=None,
        corrected_ci_low=None, corrected_ci_high=None, ceiling=None,
        difference_against_alexnet_ceiling=None, ceiling_difference_ci_low=None, ceiling_difference_ci_high=None)
    present = [i for i,m in enumerate(matrices) if m is not None]
    aligned = [matrices[i] for i in present]
    for kind, field in (("rel_a", "reliability_alexnet"), ("rel_b", "reliability_other"),
                        ("ceiling", "ceiling"), ("difference", "difference_against_alexnet_ceiling"),
                        ("corrected", "corrected_r")):
        if kind in ("rel_a", "difference") and x_retest is None:
            continue
        if kind == "rel_b" and y_retest is None:
            continue
        if kind in ("ceiling", "corrected") and (x_retest is None or y_retest is None):
            continue
        try:
            jk = PS.joint_block_jackknife(aligned, blocks, lambda *m: metric(kind, *m), unit=unit)
            out[field] = jk["estimate"]
            out[field+"_ci_low"], out[field+"_ci_high"] = jk["ci_low"], jk["ci_high"]
            if kind == "corrected":
                out.update(corrected_ci_low=jk["ci_low"], corrected_ci_high=jk["ci_high"],
                           correction="formed", correction_reason="")
            if kind == "difference":
                out.update(ceiling_difference_ci_low=jk["ci_low"], ceiling_difference_ci_high=jk["ci_high"])
        except ValueError as error:
            out["correction_reason"] = str(error)
    return out


def cross_network_rows(x, y, categories, blocks=None, x_retest=None, y_retest=None,
                       shared_indices=None, node_adjusted=False, unit="category", network="other"):
    """The raw correlation on all common categories, and the rows that need repeat extractions on the shared
    categories."""
    categories = np.asarray(categories)
    blocks = np.arange(len(categories)) if blocks is None else np.asarray(blocks)
    raw = analyze_pair_matrices(x, y, blocks, node_adjusted=node_adjusted, unit=unit)
    def row(quantity, estimate, low=None, high=None, n=None, groups=None, **extra):
        n = len(categories) if n is None else n
        return {"quantity": quantity, "network": network, "node_adjusted": node_adjusted,
            **PST.result_row(estimate, ci_low=low, ci_high=high, family="descriptive", unit=unit,
                effect_unit="correlation", scope="G" if low is not None else "D", n_categories=n,
                n_units=n if unit == "category" else len(np.unique(blocks if groups is None else groups)),
                method="joint delete-one-%s jackknife%s" %
                    (unit, "; node adjustment refitted per deletion" if node_adjusted else ""), **extra)}
    rows = [row("raw_r_all_common", raw["raw_r"], raw["ci_low"], raw["ci_high"])]
    idx = np.asarray(shared_indices if shared_indices is not None else np.arange(len(categories)), int)
    if x_retest is None and y_retest is None:
        rows.append(row("disattenuated_r_shared100", None, n=0, groups=[], status="not formed",
                        reason="one or both networks lack a declared second run"))
        return rows
    shared = analyze_pair_matrices(x[np.ix_(idx,idx)], y[np.ix_(idx,idx)], blocks[idx],
        x_retest, y_retest, node_adjusted=node_adjusted, unit=unit)
    quantities = (("reliability_alexnet_shared100", "reliability_alexnet"),
                  ("reliability_other_shared100", "reliability_other"), ("ceiling_shared100", "ceiling"),
                  ("disattenuated_r_shared100", "corrected_r"),
                  ("raw_r_minus_alexnet_ceiling_shared100", "difference_against_alexnet_ceiling"))
    for quantity, field in quantities:
        rows.append(row(quantity, shared.get(field), shared.get(field+"_ci_low"),
            shared.get(field+"_ci_high"), n=len(idx), groups=blocks[idx],
            status="formed" if shared.get(field) is not None else "not formed",
            reason="" if shared.get(field) is not None else shared["correction_reason"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ("alex-grams", "alex-pairs", "other", "taxonomy", "out"):
        ap.add_argument("--"+key, required=True)
    ap.add_argument("--alex-retest-grams"); ap.add_argument("--alex-retest-pairs")
    ap.add_argument("--other-retest"); ap.add_argument("--retest-manifest")
    ap.add_argument("--arms-config", help="the circuits' cohorts and retest pairs (arms_config.json)")
    ap.add_argument("--price", default="v66", help="which circuits (default: v66, the paper's)")
    ap.add_argument("--allow-partial", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--unit", choices=("category", "block50", "superordinate", "domain"), default="category")
    a = ap.parse_args(argv)
    if bool(a.alex_retest_grams) != bool(a.alex_retest_pairs):
        ap.error("both AlexNet retest directories are required together")
    check_outputs(files=[a.out], inputs=[a.taxonomy, a.alex_grams, a.alex_pairs, a.other,
        a.alex_retest_grams, a.alex_retest_pairs, a.other_retest, a.retest_manifest, a.arms_config])
    manifest = json.loads(Path(a.retest_manifest).read_text()) if a.retest_manifest else None
    verified_alex = bool(a.alex_retest_grams) and verify_retest_manifest(manifest, "alexnet", a.alex_grams,
        a.alex_retest_grams, a.alex_pairs, a.alex_retest_pairs)
    validate_paths([a.out], [a.taxonomy])
    tax = TS.load_tax(a.taxonomy)
    alex = {layer: excess(Path(a.alex_grams)/("gram_%s.npz" % layer),
                          Path(a.alex_pairs)/("pairs_%s.npz" % layer)) for layer in ("conv5", "fc6", "fc7")}
    config = AM.load_config(a.arms_config) if (a.alex_retest_grams or a.other_retest) else None
    rows = []
    for g in sorted(Path(a.other).glob("*__gram_*.npz")):
        net, layer = g.name.split("__gram_"); layer = layer[:-4]
        oc, oe = excess(g, Path(a.other)/("%s__pairs_%s.npz" % (net, layer)))
        for al, (ac, ae) in alex.items():
            common = np.intersect1d(ac, oc)
            ia, io = np.searchsorted(ac, common), np.searchsorted(oc, common)
            x, y = ae[np.ix_(ia,ia)], oe[np.ix_(io,io)]
            shared, xr_data, yr_data = common, None, None
            if verified_alex:
                rc, rm = excess(Path(a.alex_retest_grams)/("gram_%s.npz" % al),
                                Path(a.alex_retest_pairs)/("pairs_%s.npz" % al))
                shared, xr_data = np.intersect1d(shared, rc), (rc, rm)
            verified_other = bool(a.other_retest) and verify_retest_manifest(manifest, net, a.other, a.other_retest)
            if verified_other:
                rg, rp = Path(a.other_retest)/g.name, Path(a.other_retest)/("%s__pairs_%s.npz" % (net,layer))
                if rg.is_file() and rp.is_file():
                    rc, rm = excess(rg, rp)
                    shared, yr_data = np.intersect1d(shared, rc), (rc, rm)
            def aligned(data):
                if data is None:
                    return None
                rc, matrix = data
                ii = np.searchsorted(rc, shared)
                return matrix[np.ix_(ii,ii)]
            xr, yr = aligned(xr_data), aligned(yr_data)
            metadata, mask_verification = [], []
            for declared_network, data in (("alexnet", xr_data), (net, yr_data)):
                if data is not None:
                    pair = AM.retest_pair(config,declared_network,a.price)
                    cohort = AM.declared_cohort(config,declared_network,record={"cohort":pair["cohort"]})
                    metadata.append(AM.validate_cohort(shared.tolist(),cohort,a.allow_partial))
                    record=manifest["networks"][declared_network]
                    for name in ("primary","retest"):
                        if record["arm_"+name] != pair[name]["arm"]:
                            raise ValueError("matrix retest arm differs from arms_config named pair")
                        if record.get("run_"+name,pair[name]["run"]) != pair[name]["run"]:
                            raise ValueError("matrix retest run differs from arms_config named pair")
                    any_masks=any((AM.resolve_arm(config,declared_network,entry["arm"],a.price,
                        entry["run"],category=int(category),require_exists=False)/"final_mask.packbits").is_file()
                        for category in cohort for entry in (pair["primary"],pair["retest"]))
                    if any_masks:
                        metadata.append(AM.validate_retest_pair(config,declared_network,a.price,
                            allow_partial=a.allow_partial))
                        mask_verification.append(declared_network+": verified distinct source masks")
                    else:
                        mask_verification.append(declared_network+": not verified; source masks absent, named matrix pair only")
            partial = any(item.get("partial", False) for item in metadata)
            idx = np.searchsorted(common, shared)
            blocks = category_units(tax, common, a.unit)
            for adjusted in (False, True):
                result = cross_network_rows(x, y, common, blocks, xr, yr, idx,
                                            node_adjusted=adjusted, unit=a.unit, network=net)
                for row in result:
                    row.update(alexnet_layer=al, layer=layer, partial=partial,
                        retest_mask_verification="; ".join(mask_verification) or "not applicable: no declared second run")
                rows.extend(result)
            print(al, net, layer, "done", flush=True)
    publish_file(pd.DataFrame(rows), Path(a.out))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
