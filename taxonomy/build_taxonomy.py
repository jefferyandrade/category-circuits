#!/usr/bin/env python
"""Build the category tree, inputs/category_taxonomy.csv, from a fixed copy of the WordNet catalog of the 1,000 ImageNet
categories (inputs/category_metadata_imagenet_1000_wordnet.csv) with a cap rule on the hypernym tree.

superordinate = the groups of at most 256 categories (32 groups; the paper's superordinate level);
block50 = the groups of at most 50 categories (131 subgroups; checked against inputs/CATEGORY_BLOCKS_cap50.tsv);
depth = edges from the WordNet root. animate = the synset descends from animal.n.01 or person.n.01.

The cap rule walks down the hypernym tree from each root and splits a node into its children only while it holds
more than `cap` categories; a category whose path ends at a split node stays with that node.

Usage: python taxonomy/build_taxonomy.py [--out PATH]   (default: inputs/category_taxonomy.csv beside this script)
"""
import argparse, csv, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")


def cap_rule_blocks(paths, cap):
    out = [None] * len(paths)

    def split(d, members, node):
        if len(members) <= cap:
            for i in members:
                out[i] = node
            return
        children, ends = defaultdict(list), []
        for i in members:
            p = paths[i]
            if len(p) == d + 1:
                ends.append(i)
            else:
                children[p[d + 1]].append(i)
        for i in ends:
            out[i] = node
        for c in sorted(children):
            split(d + 1, children[c], c)

    roots = defaultdict(list)
    for i, p in enumerate(paths):
        roots[p[0]].append(i)
    for r in sorted(roots):
        split(0, roots[r], r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(INP, "category_taxonomy.csv"))
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(os.path.join(INP, "category_metadata_imagenet_1000_wordnet.csv"))))
    assert [int(r["category_id"]) for r in rows] == list(range(1000))
    paths = [r["hypernym_path"].split("|") for r in rows]
    sup = cap_rule_blocks(paths, 256)
    b50 = cap_rule_blocks(paths, 50)
    ref = {int(r["class_index"]): r
           for r in csv.DictReader(open(os.path.join(INP, "CATEGORY_BLOCKS_cap50.tsv")), delimiter="\t")}
    assert all(ref[i]["block_cap50"] == b50[i] for i in range(1000)), "the blocks differ from inputs/CATEGORY_BLOCKS_cap50.tsv"
    assert len(set(sup)) == 32 and len(set(b50)) == 131, (len(set(sup)), len(set(b50)))
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "wnid", "class_name", "animate", "dog", "superordinate", "block50", "depth"])
        for i, r in enumerate(rows):
            w.writerow([i, r["wnid"], r["class_name"], int(r["query_group"] == "animate"), int(ref[i]["dog"]), sup[i],
                        b50[i], len(paths[i]) - 1])
    print("ok: 32 superordinates, 131 cap-50 blocks; written %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
