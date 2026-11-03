"""Category-pair overlap of the kept connections on the shared categories against its size-matched hypergeometric
reference, per layer, from the shared-category incidence package. It also checks the package: each layer's
intersections by sign must add up to its intersections, and their diagonal must equal the per-category kept counts."""
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import norm
from pair_statistics import exact_J0, _pmf_window, pair_upper, jackknife_pair_mean, holm
def signmatched_J0(ap, bp, Ep, an, bn, En):
    """The sign-matched reference: I = I+ + I-, independent hypergeometric overlaps within each sign."""
    ap, bp, an, bn = int(ap), int(bp), int(an), int(bn)
    a, b = ap + an, bp + bn
    if a + b == 0: return np.nan
    if a == 0 or b == 0: return 0.0
    xp, pp = _pmf_window(Ep, ap, bp)
    xn, pn = _pmf_window(En, an, bn)
    if len(xp) == 1:   # degenerate stratum
        xs, pmf = xn + xp[0], pn
    elif len(xn) == 1:
        xs, pmf = xp + xn[0], pp
    else:
        pmf = fftconvolve(pp, pn)
        pmf = np.clip(pmf, 0, None); pmf /= pmf.sum()
        xs = np.arange(xp[0] + xn[0], xp[0] + xn[0] + len(pmf))
    denom = a + b - xs
    vals = np.where(denom > 0, xs / np.maximum(denom, 1), 1.0)
    return float((pmf * vals).sum())


def overlap(lib):
    """Per-layer overlap statistics of one network (``incidence_package.IncidencePackage``)."""
    MODEL = lib.model
    lmm, cnt = lib.layer_manifest, lib.counts
    gid, anim = np.asarray(lib.category_ids), np.asarray(lib.animate)
    K = len(gid)
    sign_split = {}
    for layer in lib.layers():
        Ip, In = lib.intersections(layer, 'pos'), lib.intersections(layer, 'neg')
        if not np.array_equal(Ip + In, lib.intersections(layer)):
            raise ValueError(f'the intersections by sign do not add up to the intersections: {MODEL}/{layer}')
        sign_split[f'{layer}|pos'] = Ip
        sign_split[f'{layer}|neg'] = In
    rows, pair_store = [], {}
    for _, lrow in lmm.iterrows():
        layer, E = lrow["layer"], int(lrow["eligible_route_count"])
        Ep, En = int(lrow["eligible_positive_count"]), int(lrow["eligible_negative_count"])
        I = lib.intersections(layer).astype(np.float64)
        a = np.diag(I).copy()
        c = cnt[(cnt.model == MODEL) & (cnt.layer == layer)].set_index("category_id")
        c = c.reindex(gid)
        assert np.array_equal(np.rint(a).astype(np.int64), c["retained_count"].to_numpy()), layer
        apos = c["retained_positive_count"].to_numpy(); aneg = c["retained_negative_count"].to_numpy()
        S = a[:, None] + a[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(S - I > 0, I / (S - I), np.nan)
        J0 = np.full((K, K), np.nan)
        J0s = np.full((K, K), np.nan)
        for k in range(K):
            for l in range(k + 1, K):
                J0[k, l] = J0[l, k] = exact_J0(a[k], a[l], E)
                J0s[k, l] = J0s[l, k] = signmatched_J0(apos[k], apos[l], Ep, aneg[k], aneg[l], En)
        ju, iu = pair_upper(J); j0u, _ = pair_upper(J0); j0su, _ = pair_upper(J0s)
        exc = ju - j0u
        with np.errstate(divide="ignore", invalid="ignore"):
            rbar = np.where(1 - j0u > 0, exc / (1 - j0u), np.nan)
        U, U_se = jackknife_pair_mean(exc, iu, K)
        R, R_se = jackknife_pair_mean(rbar, iu, K)
        Us, Us_se = jackknife_pair_mean(ju - j0su, iu, K)
        z = U / U_se if U_se > 0 else np.nan
        p_two = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
        cond = {}
        for name, mask in (("within_animate", np.outer(anim, anim)),
                           ("within_inanimate", np.outer(~anim, ~anim)),
                           ("between", np.outer(anim, ~anim) | np.outer(~anim, anim))):
            mu = mask[iu]
            vals = np.where(mu, exc, np.nan)
            m, se = jackknife_pair_mean(vals, iu, K)
            cond[name] = (m, se)
        rows.append(dict(model=MODEL, layer=layer, stage=lrow["stage"],
            eligible=E, mean_retained=float(a.mean()),
            mean_density=float((a / E).mean()),
            mean_J=float(np.nanmean(ju)), median_J=float(np.nanmedian(ju)),
            mean_J0=float(np.nanmean(j0u)), mean_J0_signmatched=float(np.nanmean(j0su)),
            U=U, U_se=U_se, z=z, p_two=p_two,
            Rbar=R, Rbar_se=R_se, U_signmatched=Us, U_signmatched_se=Us_se,
            ratio=float(np.nanmean(ju) / np.nanmean(j0u)) if np.nanmean(j0u) > 0 else np.nan,
            U_within_animate=cond["within_animate"][0], U_within_animate_se=cond["within_animate"][1],
            U_within_inanimate=cond["within_inanimate"][0], U_within_inanimate_se=cond["within_inanimate"][1],
            U_between=cond["between"][0], U_between_se=cond["between"][1],
            n_pairs_valid=int(np.isfinite(exc).sum())))
        pair_store[f"{layer}|J"] = J; pair_store[f"{layer}|J0"] = J0; pair_store[f"{layer}|J0sign"] = J0s
        print(f"[{MODEL}] {layer}: J={rows[-1]['mean_J']:.4f} J0={rows[-1]['mean_J0']:.4f} "
              f"U={U:.4f}±{U_se:.4f} ratio={rows[-1]['ratio']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_two"].to_numpy())
    return dict(stats=df, pairwise=pair_store, sign_split=sign_split)
