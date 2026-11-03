"""Weight counts and sizes by sign in AlexNet (section 2.1): are inhibitory weights fewer or weaker than excitatory ones?

Per AlexNet weight layer (conv1-fc7; checkpoint alexnet-owt-7be5be79; negative = w < 0, a zero weight counts as
positive, as in the engine): the count of each sign and the negative share of the count, |w| quantiles by sign (q05,
q25, median, q75, q90, q99), mean |w|, and each sign's share of the layer's summed |w|. Exact for the checkpoint.
Writes weights_by_sign.csv and meta.json (the checkpoint's sha256) to a new folder.

Usage: sign_magnitude.py --out NEW_DIR [--checkpoint PTH]
       sign_magnitude.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, prepare_dir, publish_file

QS = (0.05, 0.25, 0.5, 0.75, 0.9, 0.99)


def weight_table(absw: np.ndarray, neg: np.ndarray) -> dict:
    """Part (A) for one layer."""
    out = {"n_routes": int(absw.size), "n_neg": int(neg.sum()), "n_pos": int((~neg).sum())}
    tot = float(absw.sum())
    for name, sel in (("neg", neg), ("pos", ~neg)):
        a = absw[sel]
        out[f"mean_abs_{name}"] = float(a.mean()) if a.size else np.nan
        for q, v in zip(QS, np.quantile(a, QS) if a.size else [np.nan] * len(QS)):
            out[f"q{int(round(q * 100)):02d}_abs_{name}"] = float(v)
        out[f"sum_abs_share_{name}"] = float(a.sum()) / tot if tot > 0 else np.nan
    out["count_share_neg"] = out["n_neg"] / out["n_routes"]
    return out


def selftest() -> int:
    rng = np.random.default_rng(3)
    # planted: negatives smaller and more numerous
    neg = rng.random(20000) < 0.6
    absw = np.where(neg, rng.exponential(0.5, 20000), rng.exponential(1.0, 20000))
    t = weight_table(absw, neg)
    assert t["n_neg"] > t["n_pos"] and t["q50_abs_neg"] < t["q50_abs_pos"]
    assert abs(t["sum_abs_share_neg"] + t["sum_abs_share_pos"] - 1) < 1e-12
    assert abs(t["count_share_neg"] - neg.mean()) < 1e-12
    print("selftest: all checks passed")
    return 0


def run(a) -> int:
    import pandas as pd
    out = Path(a.out)
    check_outputs(dirs=[out], inputs=[a.checkpoint])
    import torch
    import circuit_masks as RT
    layout = RT.tree_layout("alexnet")
    ck = Path(a.checkpoint) if a.checkpoint else Path.home() / ".cache/torch/hub/checkpoints" / RT.CHECKPOINT_FILE["alexnet"]
    prepare_dir(out, [ck])
    negd, ck_sha = RT.weight_signs("alexnet", layout, ck)        # checks the checkpoint identity
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    wrows = []
    for s in layout.layers:
        w = sd[s.module_name + ".weight"].detach().cpu().numpy().reshape(-1).astype(np.float64)
        neg = np.asarray(negd[s.layer_id], bool).reshape(-1)
        assert w.size == s.size == neg.size and np.array_equal(neg, w < 0)
        L = RT.short_name("alexnet", s.layer_id)
        wrows.append({"layer": L, **weight_table(np.abs(w), neg)})
    publish_file(pd.DataFrame(wrows), out / "weights_by_sign.csv")
    publish_file(json.dumps({"checkpoint_sha256": ck_sha}, indent=1), out / "meta.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the checks on planted data and exit")
    ap.add_argument("--out", help="the new output folder")
    ap.add_argument("--checkpoint", help="the AlexNet checkpoint (default: the torch hub cache)")
    a = ap.parse_args(argv)
    return selftest() if a.selftest else run(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
