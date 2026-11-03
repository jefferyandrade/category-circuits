"""The layer-type tests of section 2.5: whether layer width, spatial pooling, the residual skip path or the ReLU
explains why the circuits' activations carry more category-tree structure than the prototypes only at fully connected
layers.

Each test reads one layer of one network in a different way and compares, on the 100 shared categories, the circuits'
activations with the prototypes there. As in Fig. 7, a category's circuit activation at a layer is its circuit's mean
response to the training images of all 1,000 categories, each category weighted equally, with only that layer's masks
(a residual block's, in ResNet-18) applied and the layers below dense; its prototype is the dense network's mean
response to the category's own training images. The similarity of two categories is the Pearson correlation of their
representations, and a representation's measure is the node- and permutation-adjusted superordinate tree share
(layer_statistics.tree_shares: 50 label permutations, seed 0). A reading's value is the circuits' tree share minus the
prototypes', with a delete-one-category jackknife interval: the tree shares are refitted on each deletion and the
interval is the value plus or minus a critical value times the jackknife standard error.

test         network    reading                                                        images per category
width        VGG-11     fc6 at a fixed 512 of its 4,096 units, minus conv5_2           all
pooling      VGG-11     conv5_2 position by position                                   64
skip         ResNet-18  the last block's output without the skip path                  16
before_relu  AlexNet    fc6 before the ReLU                                            all
after_relu   AlexNet    fc6 after the ReLU (Fig. 7D's AlexNet fc6 point)               all
before_relu  VGG-11     fc6 before the ReLU                                            64

Width: the 512 units, as many as conv5_2 has channels, are one fixed random draw (width_units), the same in both
representations; conv5_2 keeps its 512 channels averaged over space. The value is fc6's difference minus conv5_2's,
each category deleted from both. Pooling: the similarity of two categories is the mean, over conv5_2's 14 x 14
positions, of the correlation of their 512 channels at each position. Skip: the output of ResNet-18's last block
(layer4.1) is the ReLU of its residual branch alone, averaged over space. AlexNet fc6 before the ReLU: fc6 is affine, so
its mean response before the ReLU follows exactly from the dense network's category means of fc6's input
(before_relu_means). The AlexNet readings and the width contrast take 1.96 standard errors, as Fig. 7's AlexNet fc6 and
fc7 points do (fc_all_images.py); the others take t(.975, 99), as Fig. 7D's other points do (layer_activations.py).

Inputs:
  --activations (width)         RUN/run/vgg11 from layer_activations.py merge --out RUN (cache/COMPLETE.json,
                                cache/shared/dense.npz, shared/primary_<c>.npz), with Fig. 7D's fc6 and conv5_2 values
  --capture (pooling, skip,     the folder of a GPU pass over the first 64 (or 16) images of every category, in a
    relu-vgg11)                 fixed order: means_index.json (the shared categories, the number of images per
                                category k, and the arrays' file names) and arrays/K<k>/<site>/<arm>/<field>.npy, one
                                category-by-channel-by-position array per site, arm and field. The arm comp holds the
                                circuits' activations and image the prototypes; the field post is the site's response
                                after its ReLU, pre before it, and alpha0 ResNet-18's last block without the skip path
  --pool5 (relu-alexnet)        the dense AlexNet's category means of fc6's input over every training image
                                (category_mean, 1,000 x 9,216; image_categories, 0 to 999)
  --activations (relu-alexnet)  AlexNet's activation files of the shared categories (fc_all_images.load_run), with
                                Fig. 7D's AlexNet fc6 values; the circuits' fc6 masks come from the circuit library
                                through analysis/stats/arms_config.json, and fc6's weights from the checkpoint

Usage:
  layer_type_tests.py width --activations RUN/run/vgg11 --out width.json
  layer_type_tests.py pooling --capture CAPTURE_64 --out pooling.json
  layer_type_tests.py skip --capture CAPTURE_16 --out skip.json
  layer_type_tests.py relu-alexnet --pool5 POOL5.npz --activations ALEXNET_SHARED --out relu_alexnet.json
  layer_type_tests.py relu-vgg11 --capture CAPTURE_64 --out relu_vgg11.json
  layer_type_tests.py table width.json pooling.json skip.json relu_alexnet.json relu_vgg11.json --out DIR
The last writes DIR/layer_type_tests.tsv (one row per reading) and DIR/category_deletions.tsv (each reading's value
with each category deleted), the tables of data/layer_types/. relu-alexnet needs torch and the circuit library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

import layer_statistics as S
from layer_activations import read_taxonomy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis" / "common"))
sys.path.insert(0, str(REPO / "analysis" / "stats"))
from populations import shared100  # noqa: E402
import arms  # noqa: E402
import circuit_masks  # noqa: E402
import fc_all_images  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, prepare_dir, write_file

TAXONOMY = REPO / "taxonomy" / "inputs" / "category_taxonomy.csv"
LEVEL = "superordinate"
NORMAL_CRITICAL = 1.96
WIDTH_UNITS, WIDTH_SEED = 512, 20261023
# The images per category of the captures the paper's tests read (Methods).
CAPTURE_IMAGES = {"pooling": 64, "skip": 16, "relu-vgg11": 64}
TEST_ORDER = (("width", "vgg11"), ("pooling", "vgg11"), ("skip", "resnet18"),
              ("before_relu", "alexnet"), ("after_relu", "alexnet"), ("before_relu", "vgg11"))
TABLE_COLUMNS = ("test", "network", "layer", "reading", "images_per_category", "n_categories", "circuit",
                 "prototypes", "difference", "se", "critical_value", "ci_low", "ci_high")
DELETION_COLUMNS = ("test", "network", "deleted_category", "difference")


def pearson(values):
    """The Pearson similarity between categories of category-by-feature values (any trailing axes are flattened)."""
    values = np.asarray(values)
    result = S.pearson_similarity(values.reshape(len(values), -1))
    if result["status"] != "formed":
        raise ValueError(result["reason"])
    return result["similarity"]


def pearson_of_channel_means(values):
    """The Pearson similarity of category-by-channel-by-position values averaged over positions."""
    values = np.asarray(values)
    return pearson(values.reshape(values.shape[0], values.shape[1], -1).mean(axis=2))


def mean_pearson_over_positions(values):
    """The mean over positions of the Pearson similarity of the channel vectors at each position."""
    values = np.asarray(values)
    values = values.reshape(values.shape[0], values.shape[1], -1)
    return np.mean([pearson(values[:, :, position]) for position in range(values.shape[2])], axis=0)


def tree_share(similarity, categories, tree):
    """The node- and permutation-adjusted superordinate tree share of a category similarity matrix."""
    result = S.tree_shares(similarity, categories, tree)[LEVEL]
    if result["status"] != "formed":
        raise ValueError("the tree share is undefined: " + result["reason"])
    return result["share_adjusted"]


def deleted_shares(similarity, categories, tree):
    """The tree share with each category deleted in turn."""
    n = len(categories)
    shares = []
    for deleted in range(n):
        keep = np.arange(n) != deleted
        shares.append(tree_share(similarity[np.ix_(keep, keep)], [c for c, k in zip(categories, keep) if k], tree))
    return np.asarray(shares)


def jackknife_interval(estimate, deleted, critical):
    """The jackknife standard error of an estimate from its delete-one values, and estimate +/- critical x SE."""
    deleted = np.asarray(deleted, dtype=np.float64)
    n = len(deleted)
    se = float(np.sqrt((n - 1) / n * np.sum((deleted - deleted.mean()) ** 2)))
    return {"se": se, "critical_value": critical,
            "ci_low": estimate - critical * se, "ci_high": estimate + critical * se}


def circuits_minus_prototypes(circuit_similarity, prototype_similarity, categories, tree, critical):
    """Both tree shares, their difference, its delete-one values and its jackknife interval."""
    circuit = tree_share(circuit_similarity, categories, tree)
    prototypes = tree_share(prototype_similarity, categories, tree)
    deleted = (deleted_shares(circuit_similarity, categories, tree)
               - deleted_shares(prototype_similarity, categories, tree))
    difference = circuit - prototypes
    return dict(circuit=circuit, prototypes=prototypes, difference=difference, deleted_differences=deleted.tolist(),
                **jackknife_interval(difference, deleted, critical))


def t_critical(n_categories):
    return S.student_t_critical(n_categories - 1)


def width_units(n_units=4096):
    """fc6's fixed 512 units. The generator numpy.random.default_rng(20261023) draws twenty sets of 256 units, for a
    sensitivity check of the draw, before the 512; only the 512, sorted, are used."""
    rng = np.random.default_rng(WIDTH_SEED)
    for _ in range(20):
        rng.choice(n_units, 256, replace=False)
    return np.sort(rng.choice(n_units, WIDTH_UNITS, replace=False))


def width_contrast(fc6, conv5_2, categories, tree):
    """fc6's circuits-minus-prototypes difference at the 512 units of width_units minus conv5_2's, with the
    jackknife interval of the contrast (1.96 SE). fc6 and conv5_2 are (circuits, prototypes) pairs of category rows."""
    units = width_units(np.shape(fc6[0])[1])
    narrow = circuits_minus_prototypes(pearson(np.asarray(fc6[0])[:, units]), pearson(np.asarray(fc6[1])[:, units]),
                                       categories, tree, NORMAL_CRITICAL)
    conv = circuits_minus_prototypes(pearson(conv5_2[0]), pearson(conv5_2[1]), categories, tree, NORMAL_CRITICAL)
    difference = narrow["difference"] - conv["difference"]
    deleted = np.asarray(narrow["deleted_differences"]) - np.asarray(conv["deleted_differences"])
    return dict(circuit=None, prototypes=None, difference=difference, deleted_differences=deleted.tolist(),
                fc6_difference=narrow["difference"], conv5_2_difference=conv["difference"],
                **jackknife_interval(difference, deleted, NORMAL_CRITICAL))


def before_relu_means(pool5_means, weight, bias, categories, fc6_mask, block_rows=256):
    """AlexNet fc6 before the ReLU, from the dense network's category means of fc6's input (pool5_means, one row per
    category 0..999).

    fc6 is affine, so the mean of its pre-activation over a set of images is fc6 applied to their mean input. A
    circuit's activation is (W * M) x + b, where x is the mean over all categories of their mean inputs (each category
    weighted equally) and M the circuit's fc6 mask; a prototype is W x_c + b, where x_c is category c's mean input.
    The bias b is dense. fc6_mask(c) gives category c's mask (True where a weight is kept), in fc6's weight shape."""
    means = np.asarray(pool5_means, dtype=np.float64)
    weight, bias = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    if means.ndim != 2 or weight.shape != (len(bias), means.shape[1]) or not np.isfinite(means).all():
        raise ValueError("fc6's mean inputs, weights and bias do not fit together")
    common = means.mean(axis=0)
    circuits = np.empty((len(categories), len(bias)))
    prototypes = np.empty_like(circuits)
    for i, category in enumerate(categories):
        mask = np.asarray(fc6_mask(category))
        if mask.shape != weight.shape or mask.dtype != bool:
            raise ValueError("a circuit's fc6 mask must be boolean in fc6's weight shape")
        for first in range(0, len(bias), block_rows):
            rows = slice(first, min(first + block_rows, len(bias)))
            circuits[i, rows] = common @ (weight[rows] * mask[rows]).T + bias[rows]
            prototypes[i, rows] = means[category] @ weight[rows].T + bias[rows]
    return circuits, prototypes


def fig7_means(folder, site, categories):
    """Fig. 7D's circuit activations and prototypes at one site of a network, from the folder layer_activations.py
    merge writes (the same files its aggregate step reads)."""
    folder = Path(folder)
    record = json.loads((folder / "cache" / "COMPLETE.json").read_text())
    if record["populations"]["shared"] != list(categories):
        raise ValueError("the activations' shared categories differ from the paper's")
    with np.load(folder / "cache" / "shared" / "dense.npz", allow_pickle=False) as archive:
        if archive["categories"].tolist() != list(categories):
            raise ValueError("the prototypes' category order differs")
        prototypes = np.asarray(archive[site + "__mean"], dtype=np.float64)
    circuits = []
    for category in categories:
        with np.load(folder / "shared" / ("primary_%d.npz" % category), allow_pickle=False) as archive:
            if archive["categories"].tolist() != list(categories):
                raise ValueError("a circuit's category order differs")
            circuits.append(np.asarray(archive[site + "__mean"], dtype=np.float64))
    circuits = np.stack(circuits)
    if circuits.shape != prototypes.shape or not np.isfinite(circuits).all() or not np.isfinite(prototypes).all():
        raise ValueError("the circuits' activations and the prototypes must be finite and of one shape")
    return circuits, prototypes


def capture_means(folder, site, field, categories, images):
    """The circuits' activations and the prototypes of one site and field of a capture over the first `images` images
    of every category."""
    folder = Path(folder)
    index = json.loads((folder / "means_index.json").read_text())
    if index.get("additive") is not False:
        raise ValueError("the capture holds sums of image batches, not finished means")
    if [int(c) for c in index["category_ids"]] != list(categories):
        raise ValueError("the capture's categories differ from the paper's shared categories")
    k = int(index["k"])
    if k != images:
        raise ValueError("this test reads a capture of %d images per category; this one has %d" % (images, k))
    arms = index["subsets"]["K%d" % k][site]
    circuits = np.load(folder / arms["comp"][field], allow_pickle=False)
    prototypes = np.load(folder / arms["image"][field], allow_pickle=False)
    if (circuits.shape != prototypes.shape or len(circuits) != len(categories)
            or not np.isfinite(circuits).all() or not np.isfinite(prototypes).all()):
        raise ValueError("the circuits' activations and the prototypes must be finite and of one shape")
    return circuits, prototypes, k


def checkpoint_path(network):
    folder = Path(os.environ.get("CC_CHECKPOINTS", Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
                                 / "hub/checkpoints"))
    return folder / circuit_masks.CHECKPOINT_FILE[network]


def alexnet_fc6_weights(checkpoint):
    """fc6's weight and bias from the torchvision AlexNet checkpoint, whose sha256 must start with its file name's
    hash."""
    import torch
    expected = circuit_masks.CHECKPOINT_FILE["alexnet"].split("-")[-1].split(".")[0]
    if not hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest().startswith(expected):
        raise ValueError("%s is not the AlexNet checkpoint (%s)" % (checkpoint, expected))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return (state["classifier.1.weight"].detach().cpu().numpy(),
            state["classifier.1.bias"].detach().cpu().numpy())


def alexnet_fc6_masks(arms_config, price):
    """A function giving each category's P circuit's fc6 mask, read and checked against its receipt."""
    config = arms.load_config(arms_config)
    layout = circuit_masks.tree_layout("alexnet")
    fc6 = next(layer for layer in layout.layers if layer.module_name == "classifier.1")

    def fc6_mask(category):
        folder = arms.resolve_arm(config, "alexnet", "P", price, "primary", category=category)
        receipt = json.loads((folder / "receipt.json").read_text())
        bits = circuit_masks.load_verified_mask(folder / "final_mask.packbits", receipt, layout)
        return bits[fc6.offset:fc6.offset + fc6.size].reshape(fc6.shape)
    return fc6_mask


def row(test, network, layer, reading, images, categories, result):
    return dict(test=test, network=network, layer=layer, reading=reading, images_per_category=images,
                n_categories=len(categories), deleted_categories=list(categories), **result)


def run_width(args, categories, tree):
    fc6 = fig7_means(args.activations, "fc6", categories)
    conv5_2 = fig7_means(args.activations, "conv5_2", categories)
    return [row("width", "vgg11", "fc6", "fc6 at 512 of its 4,096 units, minus conv5_2", "all", categories,
                width_contrast(fc6, conv5_2, categories, tree))]


def run_pooling(args, categories, tree):
    circuits, prototypes, k = capture_means(args.capture, "vgg11.conv5_2", "post", categories,
                                            CAPTURE_IMAGES["pooling"])
    result = circuits_minus_prototypes(mean_pearson_over_positions(circuits), mean_pearson_over_positions(prototypes),
                                       categories, tree, t_critical(len(categories)))
    return [row("pooling", "vgg11", "conv5_2", "conv5_2 position by position", str(k), categories, result)]


def run_skip(args, categories, tree):
    circuits, prototypes, k = capture_means(args.capture, "resnet18.stage4.block1.conv2", "alpha0", categories,
                                            CAPTURE_IMAGES["skip"])
    result = circuits_minus_prototypes(pearson_of_channel_means(circuits), pearson_of_channel_means(prototypes),
                                       categories, tree, t_critical(len(categories)))
    return [row("skip", "resnet18", "stage4.block1.conv2", "the last block's output without the skip path", str(k),
                categories, result)]


def run_relu_vgg11(args, categories, tree):
    circuits, prototypes, k = capture_means(args.capture, "vgg11.fc6", "pre", categories,
                                            CAPTURE_IMAGES["relu-vgg11"])
    result = circuits_minus_prototypes(pearson(circuits), pearson(prototypes), categories, tree,
                                       t_critical(len(categories)))
    return [row("before_relu", "vgg11", "fc6", "fc6 before the ReLU", str(k), categories, result)]


def run_relu_alexnet(args, categories, tree):
    _, extracted, arrays, _ = fc_all_images.load_run(args.activations)
    if extracted != list(categories):
        raise ValueError("the AlexNet activation files must hold the shared categories")
    with np.load(args.pool5, allow_pickle=False) as archive:
        if not np.array_equal(archive["image_categories"], np.arange(1000)):
            raise ValueError("the pool5 means must hold categories 0 to 999 in order")
        pool5 = np.asarray(archive["category_mean"], dtype=np.float64)
    weight, bias = alexnet_fc6_weights(args.checkpoint or checkpoint_path("alexnet"))
    before = before_relu_means(pool5, weight, bias, categories, alexnet_fc6_masks(args.arms_config, args.price))
    after = (arrays["fc6"]["computation"]["mean"], arrays["fc6"]["image"]["mean"])
    rows = []
    for test, reading, (circuits, prototypes) in (("before_relu", "fc6 before the ReLU", before),
                                                  ("after_relu", "fc6 after the ReLU", after)):
        rows.append(row(test, "alexnet", "fc6", reading, "all", categories,
                        circuits_minus_prototypes(pearson(circuits), pearson(prototypes), categories, tree,
                                                  NORMAL_CRITICAL)))
    return rows


def write_tables(results, out):
    """layer_type_tests.tsv and category_deletions.tsv from the tests' outputs, in the order of section 2.5."""
    keys = [(r["test"], r["network"]) for result in results for r in result["rows"]]
    if sorted(keys) != sorted(TEST_ORDER):
        raise ValueError("the outputs must hold every test once: " + ", ".join("%s %s" % key for key in TEST_ORDER))
    rows = {(r["test"], r["network"]): r for result in results for r in result["rows"]}
    out = Path(out)
    prepare_dir(out)
    with write_file(out / "layer_type_tests.tsv", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        for key in TEST_ORDER:
            writer.writerow(["" if rows[key][name] is None else rows[key][name] for name in TABLE_COLUMNS])
    with write_file(out / "category_deletions.tsv", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(DELETION_COLUMNS)
        for key in TEST_ORDER:
            for category, value in zip(rows[key]["deleted_categories"], rows[key]["deleted_differences"]):
                writer.writerow([key[0], key[1], category, value])


COMMANDS = {"width": (run_width, "VGG-11 fc6 at a fixed 512 of its 4,096 units, minus conv5_2"),
            "pooling": (run_pooling, "VGG-11 conv5_2 position by position"),
            "skip": (run_skip, "ResNet-18's last block without the skip path"),
            "relu-alexnet": (run_relu_alexnet, "AlexNet fc6 before and after the ReLU"),
            "relu-vgg11": (run_relu_vgg11, "VGG-11 fc6 before the ReLU")}


def command_line():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--out", type=Path, required=True, help="the JSON to write (must not exist)")
        command.add_argument("--taxonomy", type=Path, default=TAXONOMY,
                             help="the category tree (default: taxonomy/inputs/category_taxonomy.csv)")
        if name == "width":
            command.add_argument("--activations", type=Path, required=True,
                                 help="Fig. 7D's VGG-11 activations: RUN/run/vgg11 from layer_activations.py merge --out RUN")
        if name == "relu-alexnet":
            command.add_argument("--activations", type=Path, required=True,
                                 help="Fig. 7D's AlexNet activation files of the shared categories "
                                      "(fc_all_images.py --weight)")
        if name in ("pooling", "skip", "relu-vgg11"):
            command.add_argument("--capture", type=Path, required=True,
                                 help="the capture folder (64 images per category, or 16 for skip)")
        if name == "relu-alexnet":
            command.add_argument("--pool5", type=Path, required=True,
                                 help="the dense AlexNet's category means of fc6's input")
            command.add_argument("--checkpoint", type=Path,
                                 help="the AlexNet checkpoint (default: alexnet-owt-7be5be79.pth in CC_CHECKPOINTS)")
            command.add_argument("--arms-config", type=Path,
                                 default=REPO / "analysis" / "stats" / "arms_config.json",
                                 help="where the circuits are (default: analysis/stats/arms_config.json)")
            command.add_argument("--price", default="v66", choices=("v64", "v65", "v66"),
                                 help="which circuits to read: v66, the default, is the paper's circuits")
    table = commands.add_parser("table", help="data/layer_types/'s two tables from the tests' outputs")
    table.add_argument("results", type=Path, nargs="+", help="the tests' JSON outputs")
    table.add_argument("--out", type=Path, required=True,
                       help="the folder to write the two tables in (the tables must not exist)")
    return parser


def main(argv=None):
    args = command_line().parse_args(argv)
    if args.command == "table":
        check_outputs(dirs=[args.out], inputs=args.results)
        write_tables([json.loads(path.read_text()) for path in args.results], args.out)
        return 0
    inputs = [getattr(args, name, None) for name in ("activations", "capture", "pool5", "taxonomy", "checkpoint", "arms_config")]
    check_outputs(files=[args.out], inputs=inputs)
    categories = list(shared100())
    rows = COMMANDS[args.command][0](args, categories, read_taxonomy(args.taxonomy))
    with write_file(args.out) as stream:
        json.dump({"rows": rows}, stream, indent=1, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
