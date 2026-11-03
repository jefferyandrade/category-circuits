"""Section 2.3: the Spearman correlation over layers of the wiring's tree share with the kept fraction, pooled over the
four networks, and the same with normalized depth removed (rank residuals); for all layers and for the main path
(without the ResNets' projection layers). Usage: python parting_tree_stats.py parting_tree.csv"""
import re, sys
import numpy as np
import pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1])
ALEX = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
VGG = ["conv3_1", "conv3_2", "conv4_1", "conv4_2", "conv5_1", "conv5_2", "fc6", "fc7"]
def key(net, l):
    if net == "alexnet": return ALEX.index(l)
    if net == "vgg11": return VGG.index(l)
    if l.startswith("stem"): return (0, 0, 0, 0)
    m = re.match(r"stage(\d+)\.block(\d+)\.(conv(\d+)|projection)", l)
    return (1, int(m.group(1)), int(m.group(2)), int(m.group(4)) if m.group(4) else 0)
parts = []
for net, g in d.groupby("network"):
    g = g.copy(); order = sorted(g.layer, key=lambda l: key(net, l))
    g["depth"] = [order.index(l) / (len(order) - 1) for l in g.layer]; parts.append(g)
d = pd.concat(parts); d["projection"] = d.layer.str.contains("projection")
def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return stats.pearsonr(res(rx), res(ry))
for name, sub in (("all layers", d), ("main path", d[~d.projection])):
    r, p = stats.spearmanr(sub.kept_fraction, sub.tree_share)
    rd, pd_ = stats.spearmanr(sub.depth, sub.tree_share)
    pr, pp = partial_spearman(sub.kept_fraction.values, sub.tree_share.values, sub.depth.values)
    pr2, pp2 = partial_spearman(sub.depth.values, sub.tree_share.values, sub.kept_fraction.values)
    print("%s (n=%d): tree~kept rho %.3f (p %.1e); tree~depth rho %.3f; tree~kept | depth r %.3f (p %.1e); tree~depth | kept r %.3f (p %.1e)"
          % (name, len(sub), r, p, rd, pr, pp, pr2, pp2))
m = d[~d.projection]
for lo, hi in ((0.8, 1.01), (0.6, 0.8), (0.4, 0.6), (0.0, 0.4)):
    s = m[(m.kept_fraction >= lo) & (m.kept_fraction < hi)]
    print("kept [%.1f, %.1f): n %d, tree share %.2f to %.2f, mean %.2f; networks %s" % (lo, hi, len(s), s.tree_share.min(), s.tree_share.max(), s.tree_share.mean(), dict(s.network.value_counts())))
print(d[d.projection][["network", "layer", "kept_fraction", "tree_share"]].round(3).to_string(index=False))
