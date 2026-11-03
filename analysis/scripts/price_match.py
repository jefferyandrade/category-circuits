"""Table 1 and Methods: each network's loss threshold tau ("price"), matched to AlexNet's.

Anchor: AlexNet's P circuits at tau = 1.5e-7 (they keep about 8.8% of the weights). Each other network's threshold
matches AlexNet's mean validation loss share of the class-balanced loss, (circuit - dense) / (empty - dense), where the
empty circuit's loss is ln 2. Rule: take the piloted threshold whose mean share is closest to the anchor's when it is
within one standard error of the difference; otherwise interpolate linearly in log(tau) between the two piloted
thresholds that bracket the anchor (no extrapolation).
Inputs: the threshold pilots' tables (data/thresholds/), of which the P rows with the class-balanced loss and
Firth-penalized calibration are used, under either end rule (risk_radius or walk_resolution; the masks are the same).
Usage: python3 price_match.py CSV [CSV...]"""
import csv
import math
import sys
from collections import defaultdict

rows = [r for f in sys.argv[1:] for r in csv.DictReader(open(f))
        if r["arm"] == "P" and r["weighting"] == "balanced" and r["penalty"] == "firth"]


def share(r):
    if r.get("o_final_cbb"):
        d, c, e = float(r["o_dense_cbb"]), float(r["o_final_cbb"]), float(r["o_final_empty_cbb"])
    else:
        d, c, e = float(r["cb_dense_bce"]), float(r["cb_final_bce"]), math.log(2)
    return (c - d) / (e - d)


by = defaultdict(dict)  # (network, tau) -> category -> (share, kept); first row per category wins
for r in rows:
    by[(r["network"], float(r["tau"]))].setdefault(int(r["category"]), (share(r), float(r["kept"])))


def summary(d):
    s = [v[0] for v in d.values()]
    m = sum(s) / len(s)
    se = math.sqrt(sum((x - m) ** 2 for x in s) / (len(s) - 1) / len(s))
    return m, se, 100 * sum(v[1] for v in d.values()) / len(d), len(d)


anchor_m, anchor_se, anchor_kept, anchor_n = summary(by[("alexnet", 1.5e-7)])
print(f"anchor AlexNet 1.5e-7: share {anchor_m:.4f} +/- {anchor_se:.4f}, kept {anchor_kept:.2f}%, n={anchor_n}")
for net in sorted({k[0] for k in by} - {"alexnet"}):
    pts = sorted((tau, *summary(d)) for (n, tau), d in by.items() if n == net)
    for tau, m, se, kept, n in pts:
        print(f"  {net:10s} tau {tau:.4g}: share {m:.4f} +/- {se:.4f}, kept {kept:.2f}%, n={n}")
    best = min(pts, key=lambda p: abs(p[1] - anchor_m))
    tol = math.hypot(best[2], anchor_se)
    if abs(best[1] - anchor_m) <= tol:
        print(f"  -> {net}: piloted price {best[0]:.4g} (share {best[1]:.4f}, within one SE {tol:.4f} of the anchor)")
        continue
    lo = [p for p in pts if p[1] <= anchor_m]
    hi = [p for p in pts if p[1] >= anchor_m]
    if not lo or not hi:
        print(f"  -> {net}: anchor share outside the piloted range; NO PRICE (needs another point)")
        continue
    a, b = max(lo, key=lambda p: p[1]), min(hi, key=lambda p: p[1])
    t = (anchor_m - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
    tau = math.exp(math.log(a[0]) + t * (math.log(b[0]) - math.log(a[0])))
    print(f"  -> {net}: interpolated price {tau:.4g} between {a[0]:.4g} and {b[0]:.4g}")
