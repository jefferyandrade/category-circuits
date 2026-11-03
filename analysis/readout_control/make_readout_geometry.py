"""Readout-weight geometry as a pair matrix, in the activation-archive format tree_contrast reads. At the layer that
feeds the readout, the pruning score of every connection into a unit is multiplied by |that unit's readout weight to the
target category|, so the late wiring could restate the similarity of the categories' readout weights. Writes, for site in readout_signed and readout_abs, <out>/matrices_<site>.npz with
class_ids (those of --like) and 'S|cv:all' = the cosine similarity between the categories' readout weight rows (signed,
or their absolute values), diagonal NaN as in confusion_wiring.confusability, and <out>/site_map_<site>.json mapping
every layer with a gram in --gram-dir to that site. --shuffle-seed permutes the rows across categories (planted
control: the geometry's own tree share should fall to chance).
Usage: make_readout_geometry.py --checkpoint CKPT --key KEY --like MATRICES.npz --gram-dir G --out DIR [--shuffle-seed S]
"""
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, publish_file, write_file


def cosine(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.all(n > 0):
        raise ValueError("a readout row has zero norm")
    X = X / n
    C = X @ X.T
    np.fill_diagonal(C, np.nan)
    return C


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint", "key", "like", "gram-dir", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--shuffle-seed", type=int)
    a = ap.parse_args(argv)
    prepare_dir(a.out, [a.checkpoint, a.like, a.gram_dir])
    import torch
    with np.load(a.like) as z:
        cats = np.asarray(z["class_ids"], np.int64)
    W = torch.load(a.checkpoint, map_location="cpu")[a.key].detach().double().numpy()
    if W.ndim != 2 or W.shape[0] != 1000 or cats.ndim != 1 or cats.min() < 0 or cats.max() >= W.shape[0] \
            or np.unique(cats).size != cats.size:
        raise ValueError("readout %s has shape %s; class ids %s" % (a.key, W.shape, cats[:5]))
    W = W[cats]
    if a.shuffle_seed is not None:
        W = W[np.random.default_rng(a.shuffle_seed).permutation(cats.size)]
    layers = sorted(p.stem[len("gram_"):] for p in Path(a.gram_dir).glob("gram_*.npz"))
    if not layers:
        raise ValueError("no gram_*.npz in %s" % a.gram_dir)
    out = Path(a.out)
    for kind, X in (("signed", W), ("abs", np.abs(W))):
        site = "readout_" + kind
        with write_file(out / f"matrices_{site}.npz", "wb") as stream:
            np.savez_compressed(stream, class_ids=cats, **{"S|cv:all": cosine(X)})
        publish_file(json.dumps({L: site for L in layers}, indent=1) + "\n", out / f"site_map_{site}.json")
    print("readout geometry: %s %s, %d categories, %d layers, shuffle %s -> %s"
          % (a.key, W.shape, cats.size, len(layers), a.shuffle_seed, out))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
