"""Section 2.3: does the wiring take on the category tree where circuits part?
Per layer of every network: the kept fraction (the mean over the P circuits of the gram's diagonal, divided by the
layer's number of weights) and the wiring's tree share (tree_share.py wiring: lens wiring_magnitude, set 'all',
superordinate level, node-adjusted, chance-corrected), the same quantity in every network: AlexNet on its 1,000
categories, the others on the shared 100. Writes OUT/parting_tree.csv (as in data/parting/).
Usage: python parting_tree.py STATS_OUT OUT
STATS_OUT is the statistics outputs folder (CC_STATS_OUT): AlexNet's grams are in vj_inputs/ and its tree shares in
tree_share_wiring_v66/, each other network's in xnet_<network>/."""
import glob, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, publish_file
O, OUT = sys.argv[1:]
try:
    prepare_dir(OUT, [O])
except FileExistsError as error:
    raise SystemExit(str(error))
def kept(path):
    z = np.load(path, allow_pickle=True)
    return str(z["layer"]), float(np.diag(z["I"]).mean() / int(z["n_routes"])), int(z["n_routes"]), len(z["categories"])
def shares(path):
    t = pd.read_csv(path)
    t = t[(t.level == "superordinate") & (t.lens == "wiring_magnitude") & (t.set == "all")]
    # node-adjusted, as tree_contrast.py's W
    t = t[t.node_adjusted.astype(bool)]
    assert not t.layer.duplicated().any(), path
    return t.set_index("layer")[["share_adjusted", "n_categories"]]
rows = []
src = {"alexnet": (sorted(glob.glob(O + "/vj_inputs/gram_*.npz")), O + "/tree_share_wiring_v66/tree_share_wiring.csv")}
for net in ("vgg11", "resnet18", "resnet50v1"):
    src[net] = (sorted(glob.glob(O + "/xnet_%s/other/%s__gram_*.npz" % (net, net))), O + "/xnet_%s/tree_share_wiring/tree_share_wiring_%s.csv" % (net, net))
for net, (grams, tsf) in src.items():
    ts = shares(tsf)
    if "network" in pd.read_csv(tsf).columns:
        t = pd.read_csv(tsf); t = t[(t.network == net) | (t.network.astype(str) == net)] if net == "alexnet" else t
    for g in grams:
        layer, k, n, ncat = kept(g)
        rows.append(dict(network=net, layer=layer, kept_fraction=k, n_routes=n, n_categories_gram=ncat,
                         tree_share=float(ts.share_adjusted.get(layer, np.nan)),
                         n_categories_share=ts.n_categories.get(layer, np.nan)))
d = pd.DataFrame(rows)
# every layer of every network, each with a finite share
EXPECTED = {"alexnet": 7, "vgg11": 8, "resnet18": 20, "resnet50v1": 53}
got = d.groupby("network").size().to_dict()
if got != EXPECTED or not np.isfinite(d.tree_share).all():
    raise SystemExit("incomplete layer set or missing shares: %s" % got)
publish_file(d, OUT + "/parting_tree.csv")
pd.set_option("display.width", 200)
print(d.groupby("network").size())
print(d[d.network.isin(["alexnet", "vgg11"])].round(3).to_string(index=False))
