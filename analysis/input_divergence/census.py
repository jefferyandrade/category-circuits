"""Input divergence over all of a network's extracted categories (Fig. 4C: AlexNet, all 1,000 categories).

For one layer and grain, the census reads every category's circuit and, at each group of inputs (a unit's inputs),
adds up each category pair's divergence min(c1, c2) - I and its largest possible value min(c1, c2) - max(0, c1 + c2 - n),
where c1 and c2 are the numbers of the group's n inputs that the two circuits keep and I the number they keep in common.
Only groups where the largest possible value is positive (each circuit keeps some but not all inputs) enter a pair's
sums, and a pair's input divergence is the ratio of its two sums. A category whose circuit keeps some but not all
inputs of at least one group is eligible; the layer's value is the exact mean over the defined pairs of eligible
categories (descriptive: no sampling interval), with the means within and across subgroups (WordNet blocks of at most
50 categories). From the independent repeat extractions (P-redraw) of the shared categories the census also reports
the extraction-noise floor, each category's divergence from its own repeat.

Grains: ``unit_kernel_grain`` (a convolution unit's inputs are its input channels, kept if any of their kernel
weights is kept), ``unit_fanin`` (a fully connected unit's inputs) and ``tap`` (each kernel position of a convolution
on its own). Outputs, in a new folder that appears only when complete: ``pair_totals.npz`` (the integer pair sums
``num`` and ``den`` over the P circuits followed by the P-redraw circuits, the P ``categories`` and
``redraw_categories``) and ``SUMMARY.json``. Masks are read one block of output rows at a time and never written.

One run per layer and grain of Fig. 4C (AlexNet: ``features.stage3.conv``, ``features.stage4.conv`` and
``features.stage5.conv`` at ``unit_kernel_grain``; ``readout.hidden1`` and ``readout.hidden2`` at ``unit_fanin``)::

    python analysis/input_divergence/census.py --network alexnet --layer features.stage3.conv
        --grain unit_kernel_grain --masks-root LIBRARY_RESULTS --out CENSUS/features.stage3.conv_unit_kernel_grain

(one command, shown on two lines). LIBRARY_RESULTS is the circuit library's results folder, which holds one folder
per network and category (``alexnet_tap_query<category>``). ``--p-arm`` and ``--r-arm`` default to the library's
folders of the paper's AlexNet circuits (P) and of their repeat extractions (P-redraw).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from publish import rename_new

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs

# Each network's folder-name prefix in the circuit library and its route layout (architectures.build_layout).
NETWORKS = {"alexnet": ("alexnet_tap", "alexnet_conv_kernel_weight"),
            "vgg11": ("vgg11_tap", "vgg11_conv_kernel_weight"),
            "resnet18": ("resnet18_tap", "resnet18_conv_kernel_weight")}


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

QUANTILES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def read_rows(path, spec, route_count, row0, row1):
    """Rows [row0, row1) of one layer (output units; a convolution row holds in x kh x kw weights) as booleans,
    read by bit offset from a packed big-endian circuit mask."""
    row = int(np.prod(spec.shape[1:]))
    off = int(spec.offset) + row0 * row
    size = (row1 - row0) * row
    start, stop = off // 8, (off + size + 7) // 8
    with open(path, "rb") as f:
        f.seek(start)
        raw = np.frombuffer(f.read(stop - start), dtype=np.uint8)
    if Path(path).stat().st_size != (int(route_count) + 7) // 8:
        raise ValueError(f"{path}: packed size differs from the layout")
    bits = np.unpackbits(raw, bitorder="big")[off - 8 * start: off - 8 * start + size]
    return bits.astype(bool).reshape((row1 - row0,) + tuple(spec.shape[1:]))


def to_groups(M, grain):
    """(categories, rows, ...) masks -> (categories, groups, inputs) for the grain."""
    Q, r = M.shape[:2]
    if grain == "tap":
        return M.reshape(Q, r * M.shape[2], -1)
    if grain == "unit_kernel_grain":
        return M.reshape(Q, r, M.shape[2], -1).any(-1)
    if grain == "unit_fanin":
        return M
    raise ValueError(grain)


def group_totals(B, n_max=None):
    """Exact pair sums over the groups of B, a (categories, groups, inputs) boolean array.
    
    Returns num and den, (Q, Q) int64 (symmetric, zero diagonal): the summed divergence min(a, b) - I and its
    summed largest possible value over the groups where that value is positive; and gmax, each group's largest
    pair divergence over the pairs of the first n_max categories (the P circuits; NaN where no pair is defined).
    Groups of at most 25 inputs are counted through their distinct kept patterns, larger groups through a Gram
    product per group."""
    Q, Hn, T = B.shape
    nm = Q if n_max is None else int(n_max)
    num = np.zeros((Q, Q), np.int64)
    den = np.zeros((Q, Q), np.int64)
    gmax = np.full(Hn, np.nan)
    if T <= 25:
        cnt = B.sum(-1)
        chunk = max(1, min(4096, ((1 << 24) - 1) // T, 6_000_000 // max(1, Q * T)))
        for s in range(0, Hn, chunk):
            A = B[:, s:s + chunk]
            ca = cnt[:, s:s + chunk]
            Af = A.reshape(Q, -1).astype(np.float32)
            inter = np.rint(Af @ Af.T).astype(np.int64)
            low = np.zeros((Q, Q), np.int64)
            lb = np.zeros((Q, Q), np.int64)
            for a in range(1, T + 1):
                ge = (ca >= a).astype(np.float32)
                low += np.rint(ge @ ge.T).astype(np.int64)
                lb += np.rint(ge @ (ca >= T - a + 1).astype(np.float32).T).astype(np.int64)
            num += low - inter
            den += low - lb
        weights = (1 << np.arange(T, dtype=np.int64))
        pats = (B[:nm].astype(np.int64) * weights).sum(-1)
        for h in range(Hn):
            u, mult = np.unique(pats[:, h], return_counts=True)
            bits = ((u[:, None] >> np.arange(T)) & 1).astype(bool)
            a = bits.sum(1)
            I = (bits[:, None, :] & bits[None, :, :]).sum(-1)
            mn = np.minimum(a[:, None], a[None, :])
            d = mn - np.maximum(0, a[:, None] + a[None, :] - T)
            with np.errstate(invalid="ignore", divide="ignore"):
                x = np.where(d > 0, (mn - I) / d, np.nan)
            diag = np.where((mult >= 2) & (np.diag(d) > 0), 0.0, np.nan)
            np.fill_diagonal(x, diag)
            if np.isfinite(x).any():
                gmax[h] = np.nanmax(x)
    else:
        for h in range(Hn):
            A = B[:, h].astype(np.float32)
            a = B[:, h].sum(-1).astype(np.int64)
            I = np.rint(A @ A.T).astype(np.int64)
            mn = np.minimum(a[:, None], a[None, :])
            d = mn - np.maximum(0, a[:, None] + a[None, :] - T)
            n_ = mn - I
            num += np.where(d > 0, n_, 0)
            den += np.where(d > 0, d, 0)
            ium = np.triu_indices(nm, 1)
            ok = d[ium] > 0
            if ok.any():
                gmax[h] = float((n_[ium][ok] / d[ium][ok]).max())
    np.fill_diagonal(num, 0)
    np.fill_diagonal(den, 0)
    return num, den, gmax


def exact_summary(X, blocks, num):
    """Exact summaries of a symmetric pair matrix X over all unordered pairs of the given categories.
    
    With every pair defined: the mean over all pairs, within and across blocks, the share of pairs with any
    divergence, quantiles and the largest pair value. With undefined pairs the complete summaries are not formed,
    and the mean over the defined pairs is reported under its own name."""
    Q = X.shape[0]
    iu = np.triu_indices(Q, 1)
    x = X[iu]
    ok = np.isfinite(x)
    same = (blocks[:, None] == blocks[None, :])[iu]
    out = {"n_categories": int(Q), "n_pairs": int(len(x)), "n_pairs_defined": int(ok.sum()),
           "label": "exact over the extracted categories (descriptive)"}
    if len(x) == 0:
        out["status"] = "not formed"
        out["reason"] = "fewer than two categories"
        return out
    if not ok.all():
        out.update({"status": "not formed", "reason": f"{int((~ok).sum())} undefined pairs",
                    "mean_over_defined_pairs_only": float(x[ok].mean()) if ok.any() else None,
                    "mean_all": None})
        return out
    out.update({"mean_all": float(x.mean()),
                "mean_cross_block": float(x[~same].mean()) if (~same).any() else None,
                "mean_within_block": float(x[same].mean()) if same.any() else None,
                "n_within_block": int(same.sum()),
                "share_pairs_with_any_divergence": float((num[iu] > 0).mean()),
                "quantiles": dict(zip(map(str, QUANTILES), np.quantile(x, QUANTILES).tolist())),
                "max_pair": float(x.max())})
    return out


def gmax_summary(g):
    """Summaries of the groups' largest pair divergences."""
    ok = np.isfinite(g)
    return {"n_groups": int(len(g)), "n_groups_with_feasible_pair": int(ok.sum()),
            "share_groups_max_positive": float((g[ok] > 0).mean()) if ok.any() else None,
            "share_groups_max_one": float((g[ok] == 1).mean()) if ok.any() else None,
            "mean_of_group_max": float(g[ok].mean()) if ok.any() else None,
            "quantiles_of_group_max": dict(zip(map(str, QUANTILES), np.quantile(g[ok], QUANTILES).tolist()))
            if ok.any() else None}


def pair_x(num, den):
    """Pair values num / den; NaN where den is zero and on the diagonal."""
    with np.errstate(invalid="ignore", divide="ignore"):
        X = np.where(den > 0, num / den, np.nan)
    np.fill_diagonal(X, np.nan)
    return X


def run(args):
    """Compute one layer and grain of the census and publish its output folder; returns its path."""
    out = Path(args.out)
    check_outputs(dirs=[out], inputs=[args.masks_root])
    if out.exists():
        out.rmdir()
    from populations import labels, shared100
    from architectures import build_layout
    prefix, family = NETWORKS[args.network]
    layout = build_layout(family)
    spec = layout.layer(args.layer)
    root = Path(args.masks_root)
    present = sorted(int(p.name.split("query")[1]) for p in root.glob(f"{prefix}_query*")
                     if (p / args.p_arm / "final_mask.packbits").is_file())
    cats = [c for c in present if c < 1000]
    if args.require_all and len(cats) != 1000:
        raise ValueError(f"{len(cats)} of 1000 categories have a P mask")
    s100 = set(shared100())
    r_cats = [c for c in sorted(s100) if (root / f"{prefix}_query{c}" / args.r_arm / "final_mask.packbits").is_file()] \
        if args.r_arm else []
    paths = [root / f"{prefix}_query{c}" / args.p_arm / "final_mask.packbits" for c in cats] + \
            [root / f"{prefix}_query{c}" / args.r_arm / "final_mask.packbits" for c in r_cats]
    Q = len(paths)
    rows = spec.shape[0]
    rowbytes = int(np.prod(spec.shape[1:])) * Q
    chunk = max(1, min(rows, int(args.chunk_bytes) // max(1, rowbytes)))
    num = np.zeros((Q, Q), np.int64)
    den = np.zeros((Q, Q), np.int64)
    gmaxes = []
    partial = np.zeros(Q, dtype=bool)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        M = np.stack([read_rows(p, spec, layout.route_count, r0, r1) for p in paths])
        G_ = to_groups(M, args.grain)
        a_ = G_.sum(-1)
        partial |= ((a_ > 0) & (a_ < G_.shape[-1])).any(axis=1)
        n_, d_, g_ = group_totals(G_, n_max=len(cats))
        num += n_
        den += d_
        gmaxes.append(g_)
        del M
    gmax_all = np.concatenate(gmaxes)
    nP = len(cats)
    X_all = pair_x(num, den)
    eligP = partial[:nP]
    elig_cats = [c for c, e in zip(cats, eligP) if e]
    XP = X_all[:nP, :nP][np.ix_(eligP, eligP)]
    blocksP = labels(elig_cats)
    summary = {"network": args.network, "layer": args.layer,
               "grain": args.grain, "P_arm": args.p_arm, "R_arm": args.r_arm or None,
               "n_P": nP, "n_R_shared100": len(r_cats), "argv": sys.argv,
               "n_P_ineligible": int((~eligP).sum()),
               "ineligible_categories": [c for c, e in zip(cats, eligP) if not e],
               "exact_P": exact_summary(XP, blocksP, num[:nP, :nP][np.ix_(eligP, eligP)])}
    summary["group_max_over_pairs_P"] = gmax_summary(gmax_all)
    if r_cats:
        idxP = {c: i for i, c in enumerate(cats)}
        rpos = {c: nP + k for k, c in enumerate(r_cats)}
        avail = [c for c in r_cats if c in idxP]
        shared = [c for c in avail if partial[idxP[c]] and partial[rpos[c]]]
        ip = np.array([idxP[c] for c in shared], dtype=np.int64)
        ir = np.array([rpos[c] for c in shared], dtype=np.int64)
        floor = np.array([X_all[x, y] for x, y in zip(ip, ir)], dtype=float)
        bS = labels(shared) if shared else np.zeros(0, np.int64)
        fl = {"n": len(shared), "n_available": len(avail),
              "excluded_ineligible_in_an_extraction": [c for c in avail if c not in shared]}
        if len(floor) and np.isfinite(floor).all():
            fl.update({"floor_mean": float(floor.mean()), "floor_max": float(floor.max()),
                       "floor_n_positive": int(np.sum(floor > 0))})
        else:
            fl.update({"floor_mean": None, "floor_max": None, "floor_n_positive": int(np.sum(floor > 0)),
                       "floor_status": "not formed" if len(floor) else "no categories",
                       "floor_reason": f"{int(np.sum(~np.isfinite(floor)))} undefined floors"})
        fl["exact_PP_shared100"] = exact_summary(X_all[np.ix_(ip, ip)], bS, num[np.ix_(ip, ip)])
        fl["exact_RR_shared100"] = exact_summary(X_all[np.ix_(ir, ir)], bS, num[np.ix_(ir, ir)])
        sub = np.arange(nP)
        for c in avail:
            sub[idxP[c]] = rpos[c]
        common = partial[:nP] & partial[sub]
        cs_ = [c for c, e in zip(cats, common) if e]
        bC = labels(cs_) if cs_ else np.zeros(0, np.int64)
        iP = np.flatnonzero(common)
        iS = sub[common]
        fl["common_set_n"] = int(common.sum())
        fl["exact_all_P_on_common_set"] = exact_summary(X_all[np.ix_(iP, iP)], bC, num[np.ix_(iP, iP)])
        fl["exact_all_with_redraws_substituted"] = exact_summary(X_all[np.ix_(iS, iS)], bC, num[np.ix_(iS, iS)])
        summary["extraction_noise_shared100"] = fl
    np_tmp = out.parent / f".tmp-{out.name}.{os.getpid()}"
    np_tmp.mkdir(parents=True)
    np.savez_compressed(np_tmp / "pair_totals.npz", num=num, den=den, categories=np.array(cats),
                        redraw_categories=np.array(r_cats), group_max=gmax_all)
    summary["inputs_sha256"] = {str(p): sha256_file(p) for p in paths} if args.hash_inputs else "not hashed"
    (np_tmp / "SUMMARY.json").write_text(json.dumps(_clean(summary), indent=1, allow_nan=False) + "\n")
    rename_new(np_tmp, out)
    return str(out)


def parser():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--network", choices=sorted(NETWORKS), required=True)
    p.add_argument("--layer", required=True, help="the layer's name in the route layout, e.g. readout.hidden1")
    p.add_argument("--grain", choices=("tap", "unit_kernel_grain", "unit_fanin"), required=True)
    p.add_argument("--out", required=True, help="output folder (must not exist)")
    p.add_argument("--masks-root", required=True, help="the circuit library's results folder")
    p.add_argument("--p-arm", default="v0_local_attribution_sign_plain_t1p5e7_v64",
                   help="circuit-library folder of the P circuits (default: the paper's AlexNet circuits)")
    p.add_argument("--r-arm", default="v0_local_attribution_sign_plain_p_redraw_t1p5e7_v64",
                   help="folder of the P-redraw repeat extractions of the shared categories ('' for none)")
    p.add_argument("--require-all", action="store_true", help="stop unless all 1,000 categories have a P circuit")
    p.add_argument("--chunk-bytes", type=float, default=2.0e9,
                   help="bytes of mask rows held in memory at a time (changes no result)")
    p.add_argument("--hash-inputs", action="store_true", help="record the sha256 of every mask read in SUMMARY.json")
    return p


def main(argv=None):
    """Run the census with one numerical-library thread."""
    args = parser().parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        print(run(args))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
