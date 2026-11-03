#!/usr/bin/env python3
"""AlexNet's input-divergence references on all 1,000 categories (Fig. 4C, AlexNet's dashes and differences).

For conv3, conv4, conv5, fc6 and fc7, this script reads the census's integer pair sums (one folder per layer and
grain below CC_CENSUS, as ``census.py --out`` wrote them), recomputes each circuit's kept inputs per unit and per
decile of input weight size from its mask and AlexNet's checkpoint, and writes values_fig4C_reference.csv: the
observed mean divergence, the uniform and weight-size-matched references, the observed minus each reference with
paired delete-one-category jackknife intervals over the eligible categories, the number of defined pairs, and the
census's extraction-noise floor and within- and cross-subgroup means. The masks are those the census read (its
command line names the library, and its summary the P circuits' folder). Three checks tie the two readings together:
the maxima recomputed from the masks must equal the census's denominators, the eligible categories the census's, and
the observed mean the census summary's. values_fig4C_reference.json records the checkpoint and census files read,
with their sha256, and each layer's jackknife results.

    CC_CENSUS=CENSUS python analysis/input_divergence/alexnet_references.py --out REFERENCES [--layer fc6]

Needs torch and the torchvision AlexNet checkpoint.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, prepare_dir, publish_file, write_file

# The AlexNet census of all 1,000 categories: one folder per layer and grain (census.py --out).
CENSUS = Path(os.environ.get('CC_CENSUS', '.'))
LAYERS = [('conv3', 'features.stage3.conv', 'unit_kernel_grain'),
          ('conv4', 'features.stage4.conv', 'unit_kernel_grain'),
          ('conv5', 'features.stage5.conv', 'unit_kernel_grain'),
          ('fc6', 'readout.hidden1', 'unit_fanin'),
          ('fc7', 'readout.hidden2', 'unit_fanin')]


def read_rows(path, spec, route_count, row0, row1):
    """Rows [row0, row1) of one layer from a packed big-endian circuit mask."""
    row = int(np.prod(spec.shape[1:]))
    off, size = int(spec.offset) + row0 * row, (row1 - row0) * row
    start, stop = off // 8, (off + size + 7) // 8
    if Path(path).stat().st_size != (int(route_count) + 7) // 8:
        raise ValueError('packed mask size differs from layout: %s' % path)
    with open(path, 'rb') as f:
        f.seek(start)
        raw = np.frombuffer(f.read(stop - start), dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder='big')[off - 8 * start:off - 8 * start + size]
    return bits.astype(bool).reshape((row1 - row0,) + tuple(spec.shape[1:]))


def decile_labels(scores):
    """Ten rank deciles: floor(10 r / n), ties in input order; one stratum below ten inputs."""
    h, n = scores.shape
    labels = np.zeros((h, n), dtype=np.uint8)
    if n >= 10:
        order = np.argsort(scores, axis=1, kind='stable')
        np.put_along_axis(labels, order, np.broadcast_to((10 * np.arange(n) // n).astype(np.uint8), (h, n)), axis=1)
    return labels


def input_scores(weight):
    """Input weight sizes: |w| for fully connected weights, a kernel's summed |w| for convolutions."""
    weights = weight.detach().cpu().numpy()
    return np.abs(weights).sum(axis=(2, 3)) if weights.ndim == 4 else np.abs(weights)


def reference_totals(counts, strata, sizes, return_den=False):
    """Pair sums of the uniform and weight-size-matched references (and the maximum divergence) from
    kept counts per unit and per unit and stratum."""
    q, h = counts.shape
    n = sizes.sum(axis=1)
    low = np.zeros((q, q), dtype=np.int64)
    den = np.zeros_like(low)
    for k in range(h):
        a = counts[:, k].astype(np.int64)
        mn = np.minimum(a[:, None], a[None, :])
        low += mn
        den += mn - np.maximum(0, a[:, None] + a[None, :] - n[k])
    z = counts.astype(np.float64) / np.sqrt(n)[None, :]
    uniform = low - z @ z.T
    z = np.divide(strata, np.sqrt(sizes)[None, :, :],
                  out=np.zeros(strata.shape, dtype=np.float64), where=sizes[None, :, :] > 0).reshape(q, -1)
    matched = low - z @ z.T
    if return_den:
        np.fill_diagonal(den, 0)
        return uniform, matched, den
    return low, uniform, matched


def pair_values(num, den):
    """Pair ratios; undefined pairs and the diagonal are NaN."""
    result = np.full(den.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=result, where=den > 0)
    np.fill_diagonal(result, np.nan)
    return result


def check_observed(actual, recorded):
    """The observed mean must equal the census summary's value exactly (it prints the full binary64 value)."""
    if actual != recorded:
        raise AssertionError('the observed mean %r differs from the census summary\'s %r' % (actual, recorded))


def jackknife(x, critical=None):
    """Mean over defined pairs with a delete-one-category jackknife interval, t(0.975, Q - 1)."""
    q = len(x)
    iu = np.triu_indices(q, 1)
    vals = x[iu]
    ok = np.isfinite(vals)
    total, count = vals[ok].sum(), int(ok.sum())
    row_sum = np.nansum(x, axis=1)
    row_n = np.isfinite(x).sum(axis=1)
    deleted = (total - row_sum) / (count - row_n)
    if not np.isfinite(deleted).all():
        raise ValueError('delete-category replicate has no defined pairs')
    point = float(total / count)
    se = float(np.sqrt((q - 1.) / q * np.square(deleted - deleted.mean()).sum()))
    if critical is None:
        from scipy.stats import t
        critical = float(t.ppf(.975, q - 1))
    return dict(point=point, se=se, lower=point-critical*se, upper=point+critical*se,
                critical=critical, n_categories=q)


def sha256(path):
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    """Compute the references of every layer and write values_fig4C_reference.csv and values_fig4C_reference.json."""
    output = Path(args.out)
    sources = [CENSUS / (layer + '_' + grain) / filename
               for name, layer, grain in LAYERS if not args.layer or name == args.layer
               for filename in ('SUMMARY.json', 'pair_totals.npz')]
    check_outputs(dirs=[output], inputs=sources)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
    import torch
    from architectures import CHECKPOINT_DIR, build_layout, checkpoint_path
    from census import NETWORKS
    torch.set_num_threads(1)
    layout = build_layout('alexnet_conv_kernel_weight')
    checkpoint = checkpoint_path('alexnet')
    if checkpoint is None:
        raise FileNotFoundError('the torchvision AlexNet checkpoint is not in ' + str(CHECKPOINT_DIR))
    prepare_dir(output, [*sources, checkpoint])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    rows, detail = [], []
    for name, layer, grain in LAYERS:
        if args.layer and name != args.layer:
            continue
        start = time.perf_counter()
        census = CENSUS / (layer + '_' + grain)
        summary_path = census / 'SUMMARY.json'
        summary = json.loads(summary_path.read_text())
        with np.load(census / 'pair_totals.npz') as packed:
            cats = packed['categories'].copy()
            q = len(cats)
            observed_num = packed['num'][:q, :q].copy()
            den = packed['den'][:q, :q].copy()
        spec = layout.layer(layer)
        scores = input_scores(state[spec.module_name + '.weight'])
        labels = decile_labels(scores)
        h, n = labels.shape
        ns = 1 if n < 10 else 10
        sizes = np.stack([(labels == s).sum(axis=1) for s in range(ns)], axis=1)
        # (unit, stratum) ids fit uint16 for AlexNet's at most 4,096 units and ten strata.
        ids = (np.arange(h, dtype=np.uint32)[:, None] * ns + labels).astype(np.uint16)
        counts = np.empty((q, h), dtype=np.uint16)
        strata = np.empty((q, h, ns), dtype=np.uint16)
        input_hash = hashlib.sha256()
        root = Path(summary['argv'][summary['argv'].index('--masks-root') + 1])
        for i, cat in enumerate(cats):
            path = root / ('%s_query%d' % (NETWORKS['alexnet'][0], cat)) / summary['P_arm'] / 'final_mask.packbits'
            mask = read_rows(path, spec, layout.route_count, 0, h)
            if mask.ndim == 4:
                mask = mask.any(axis=(2, 3))
            sc = np.bincount(ids[mask], minlength=h * ns).reshape(h, ns)
            strata[i] = sc
            counts[i] = sc.sum(axis=1)
            input_hash.update(str(int(cat)).encode() + b':' + np.packbits(mask, bitorder='big').tobytes())
            if (i + 1) % 100 == 0:
                print('%s counts %d/%d %.3fs' % (name, i + 1, q, time.perf_counter()-start), flush=True)
        uniform, matched, calculated_den = reference_totals(counts, strata, sizes, return_den=True)
        if not np.array_equal(den, calculated_den):
            raise AssertionError('the maxima recomputed from the masks differ from the census\'s denominators')
        eligible = ((counts > 0) & (counts < n)).any(axis=1)
        bad = cats[~eligible].tolist()
        if bad != summary['ineligible_categories']:
            raise AssertionError('the eligible categories differ from the census\'s')
        ix = np.ix_(eligible, eligible)
        observed = pair_values(observed_num[ix], den[ix])
        u, m = pair_values(uniform[ix], den[ix]), pair_values(matched[ix], den[ix])
        iu = np.triu_indices(len(observed), 1)
        observed_values = observed[iu]
        obs = float(observed_values[np.isfinite(observed_values)].mean())
        exact = summary['exact_P']
        recorded = exact['mean_all'] if exact['mean_all'] is not None else exact['mean_over_defined_pairs_only']
        check_observed(obs, recorded)
        ju, jm = jackknife(observed-u), jackknife(observed-m)
        row = dict(layer=name, observed=obs, uniform_reference=float(np.nanmean(u[iu])),
                   weight_size_matched_reference=float(np.nanmean(m[iu])),
                   observed_minus_uniform=ju['point'], uniform_difference_lower=ju['lower'], uniform_difference_upper=ju['upper'],
                   observed_minus_weight_size_matched=jm['point'], weight_size_difference_lower=jm['lower'], weight_size_difference_upper=jm['upper'],
                   n_categories=len(observed), n_pairs_defined=int(np.isfinite(observed[iu]).sum()),
                   extraction_noise_floor=summary['extraction_noise_shared100']['floor_mean'],
                   within_block_mean=exact.get('mean_within_block'), cross_block_mean=exact.get('mean_cross_block'),
                   block_means_status='present' if 'mean_cross_block' in exact else 'not formed in the census (undefined pairs)',
                   census_summary=str(summary_path), decile_binning='stable ascending ranks; floor(10*rank/n); input-index ties; n<10: one stratum',
                   interval='paired delete-one-category jackknife; point +/- t(.975,Q-1)*SE; defined pairs only')
        rows.append(row)
        item = dict(layer=name, mask_membership_sha256=input_hash.hexdigest(),
                    census_summary_sha256=sha256(summary_path), pair_totals_sha256=sha256(census / 'pair_totals.npz'),
                    jackknife_uniform=ju, jackknife_matched=jm)
        detail.append(item)
        print(json.dumps(item), flush=True)
    with write_file(output / 'values_fig4C_reference.csv', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), layers=detail,
                  python=sys.version, numpy=np.__version__, torch=torch.__version__)
    publish_file(json.dumps(record, indent=2)+'\n', output / 'values_fig4C_reference.json')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
        parser.add_argument('--out', required=True, help='output folder')
        parser.add_argument('--layer', choices=[x[0] for x in LAYERS], help='one layer only (default: all five)')
        run(parser.parse_args())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
