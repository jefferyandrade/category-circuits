"""The output-geometry control of section 2.4: the dense network's output geometry as a pair matrix, in the format
tree_share.py, tree_contrast.py and cross_network_adjusted.py read for the mean category representations. Each circuit
preserves its category's output, so the wiring is compared with the output geometry it is built to preserve.

Writes <out>/matrices_logits.npz with class_ids and 'S|cv:all' = the symmetrized dense mean-logit matrix
(confusion_wiring.confusability), and <out>/site_map_logits.json, which maps every layer with a gram in --gram-dir to
the site 'logits'. tree_contrast.py, given <out> as --trained-dir with that site map, then gives the wiring's tree
share beyond the outputs (W|A).
Usage: make_logit_geometry.py --category-matrix category_matrix_dense.npz --gram-dir G --out DIR
"""
import argparse, io, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from confusion_wiring import confusability  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_files, prepare_dir  # noqa: E402

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("category-matrix", "gram-dir", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    prepare_dir(out, [a.category_matrix, a.gram_dir])
    cats, C = confusability(a.category_matrix)
    buf = io.BytesIO(); np.savez_compressed(buf, class_ids=cats, **{"S|cv:all": C})
    layers = sorted(p.stem[len("gram_"):] for p in Path(a.gram_dir).glob("gram_*.npz"))
    publish_files([(buf.getvalue(), out / "matrices_logits.npz"),
                   (json.dumps({L: "logits" for L in layers}, indent=1) + "\n", out / "site_map_logits.json")])
    print("logit geometry: %d categories, %d layers -> %s" % (cats.size, len(layers), out))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
