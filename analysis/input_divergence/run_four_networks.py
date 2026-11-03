#!/usr/bin/env python3
"""Input divergence of the four networks on the shared categories (Fig. 4C, VGG-11 and the ResNets).

For every pruned layer of AlexNet, VGG-11, ResNet-18 and ResNet-50 on the shared categories (100, or 99 in
ResNet-50), this script reads the circuits' masks and the checkpoint's weights (``network_sources.py``) and computes,
with ``divergence_math.py``:

  observed      the mean input divergence over the pairs where it is defined, with a delete-one-category
                jackknife interval;
  ref_uniform   the mean expected divergence if each circuit chose its kept inputs of a unit at random;
  ref_weight    the same within each decile of input weight size (the dashes of Fig. 4C);
  obs_minus_*   the observed minus each reference, with paired jackknife intervals;
  flagged_omit  a layer where fewer than half of the pairs are defined (left out of the figure).

Two modes:

  catalog  writes the table of sources (--catalog) from two tables in --workspace: ``u_xnet.csv``, the source table
           of the overlap runs (``analysis/stats/overlap_nulls.py``; each layer's overlap summary, beside whose
           ``meta.json`` lie the circuit library's folder, the P circuits' folder and the checkpoint), and
           ``xnet_sign_density.tsv``, the kept fractions by sign of each network and layer. AlexNet's layers point at
           its census (CC_CENSUS, one folder per layer and grain, as ``census.py --out`` wrote them).
  full     computes every layer and writes values_fig4C_four_networks.csv (one row per network and layer) in --out,
           with each layer's pair sums in pair_totals/ and its values in layers/ (a layer whose file is there is not
           computed again).

    python analysis/input_divergence/run_four_networks.py --mode catalog --workspace SOURCES --catalog CATALOG.json
    python analysis/input_divergence/run_four_networks.py --mode full --catalog CATALOG.json --out OUT

Every mask is checked against its receipt and its layer's kept count against the overlap run's Gram files and the
recorded kept fractions by sign, and each network's kept percentage must round to Table 1's. Needs torch and the
torchvision checkpoints.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_file, write_file, validate_paths

# The AlexNet census of all 1,000 categories: one folder per layer and grain (census.py --out).
CENSUS = Path(os.environ.get('CC_CENSUS', '.'))
ALEX = [('conv1', 'features.stage1.conv', 'unit_kernel_grain'),
        ('conv2', 'features.stage2.conv', 'unit_kernel_grain'),
        ('conv3', 'features.stage3.conv', 'unit_kernel_grain'),
        ('conv4', 'features.stage4.conv', 'unit_kernel_grain'),
        ('conv5', 'features.stage5.conv', 'unit_kernel_grain'),
        ('fc6', 'readout.hidden1', 'unit_fanin'),
        ('fc7', 'readout.hidden2', 'unit_fanin')]
NETWORKS = ['alexnet', 'vgg11', 'resnet18', 'resnet50v2']
# Each network's kept percentage as Table 1 prints it: a check that the masks read are the paper's.
TABLE1_KEPT_PERCENT = {'alexnet': '8.9', 'vgg11': '5.0', 'resnet18': '27.2', 'resnet50v2': '31.4'}
N_LAYERS = 88


def build_catalog(workspace):
    """The sources of every network and layer, read from the supplied tables (never inferred from run names)."""
    workspace = Path(workspace)
    networks = {name: {'layers': [], 'densities': {}} for name in NETWORKS}
    seen = set()
    path = workspace / 'u_xnet.csv'
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            name, layer = row['network'], row['layer']
            if (name, layer) in seen:
                continue
            seen.add((name, layer))
            networks[name]['layers'].append(dict(layer=layer, source=row['source'],
                meta=str(Path(row['source']).parent / 'meta.json')))
    for name, lid, grain in ALEX:
        source = CENSUS / (lid + '_' + grain) / 'SUMMARY.json'
        if name == 'conv1':
            # conv1 has no census of its own; conv3's census names the same mask library and folder.
            source = CENSUS / 'features.stage3.conv_unit_kernel_grain/SUMMARY.json'
        networks['alexnet']['layers'].append(dict(layer=name, layer_id=lid,
            source=str(source), meta=None))
    with (workspace / 'xnet_sign_density.tsv').open(newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['network'] in networks:
                networks[row['network']]['densities'][row['layer']] = {
                    key: row[key] for key in ['dens_pos', 'dens_neg']}
    return dict(networks=networks, overlap_source=str(path),
        census_source=str(CENSUS),
        n_categories=dict(alexnet=100, vgg11=100, resnet18=100, resnet50v2=99))


def check_printed(actual, printed):
    """Check that a value rounds to a printed number at the printed precision."""
    places = len(printed.split('.')[1]) if '.' in printed else 0
    if ('%.*f' % (places, actual)) != printed:
        raise AssertionError('%r does not round to the printed %s' % (actual, printed))


def atomic_json(path, data):
    """Publish one completed JSON record without replacing an earlier record."""
    publish_file(json.dumps(data, indent=2, allow_nan=True) + '\n', path)


def file_sha(path):
    """sha256 of a file."""
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def census_p_values(raw):
    """Pair values of the P circuits from a census pair_totals.npz: the census appends the P-redraw circuits
    after the P categories, so the leading [:Q, :Q] block is selected."""
    import numpy as np
    from divergence_math import pair_values
    categories = list(map(int, raw['categories']))
    q = len(categories)
    num, den = np.asarray(raw['num']), np.asarray(raw['den'])
    if len(set(categories)) != q or q < 2:
        raise ValueError('census P categories must be distinct and nonempty')
    if num.ndim != 2 or num.shape != den.shape or num.shape[0] != num.shape[1] or num.shape[0] < q:
        raise ValueError('census totals must be matching square matrices containing every P category')
    return pair_values(num[:q,:q], den[:q,:q])


def layer_task(ctx, item, output):
    """One layer of one network: pair sums over unit batches and the summary values."""
    import numpy as np
    from divergence_math import PairTotals, decile_labels, summarize
    from network_sources import iter_input_chunks
    cats = list(map(int, ctx['categories']))
    q = len(cats)
    spec = item['spec']
    # Units per batch bound the category-by-input boolean array to about 16 MiB (a resource setting).
    unit_chunk = max(1, min(int(spec.shape[0]), (16 * 1024 * 1024) // (q * int(spec.shape[1]))))
    totals = PairTotals(q, category_ids=cats)
    for mask, scores in iter_input_chunks(ctx, item, cats, unit_chunk):
        totals.update(mask, decile_labels(scores), category_ids=cats)
    raw = totals.finalize()
    values = summarize(raw)
    record = dict(values)
    record.update(network=ctx['network'], layer=item['layer'],
        layer_index=item['layer_index'], n_layers=len(ctx['layers']), n_categories=q,
        flagged_omit=(2 * values['n_pairs_defined'] < values['n_pairs']),
        source=item['source'])
    result = dict(values=record, kept=item['kept'],
        mask_root=str(ctx['mask_root']), arm=ctx['arm'], checkpoint=str(ctx['checkpoint']),
        checkpoint_sha256=ctx.get('checkpoint_sha256'), category_ids=cats)
    raw_path = Path(output) / 'pair_totals' / (ctx['network'] + '__' + item['layer'] + '.npz')
    if not raw_path.exists():
        with write_file(raw_path, 'wb') as stream:
            np.savez_compressed(stream, **raw, categories=np.asarray(cats))
    else:
        with np.load(raw_path, allow_pickle=False) as saved:
            expected = dict(raw, categories=np.asarray(cats))
            if set(saved.files) != set(expected) or any(not np.array_equal(saved[k], v, equal_nan=True)
                                                       for k, v in expected.items()):
                raise ValueError('stored pair totals differ: %s' % raw_path)
    result['pair_totals_path'] = str(raw_path)
    result['pair_totals_sha256'] = file_sha(raw_path)
    return result


FIELDS = ['network','layer','layer_index','n_layers','n_categories','observed','ci_low','ci_high',
          'ref_uniform','ref_weight','obs_minus_uniform','obs_minus_uniform_ci_low','obs_minus_uniform_ci_high',
          'obs_minus_weight','obs_minus_weight_ci_low','obs_minus_weight_ci_high','n_pairs_defined','n_pairs',
          'mean_partial_share','flagged_omit','source']


def write_csv(path, rows, fields):
    """Publish the completed table without replacing an earlier table."""
    with write_file(path, newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def assemble(output):
    """Collect the per-layer values into values_fig4C_four_networks.csv; returns the number of rows and the
    flagged layers."""
    output = Path(output)
    check_outputs(files=[output / 'values_fig4C_four_networks.csv'])
    records = [json.loads(path.read_text()) for path in sorted((output / 'layers').glob('*.json'))]
    shared = [row['values'] for row in records]
    order = {name:i for i,name in enumerate(NETWORKS)}
    shared.sort(key=lambda row:(order[row['network']], row['layer_index']))
    write_csv(output / 'values_fig4C_four_networks.csv', shared, FIELDS)
    flags = [row['network']+':'+row['layer'] for row in shared if row['flagged_omit']]
    return dict(rows=len(shared), flagged=flags)


def run(args):
    """Run the selected mode."""
    if args.mode == 'catalog':
        sources = [Path(args.workspace) / 'u_xnet.csv', Path(args.workspace) / 'xnet_sign_density.tsv']
        check_outputs(files=[args.catalog], inputs=sources)
        atomic_json(args.catalog, build_catalog(args.workspace))
        print('Source catalog written: 88 layers')
        return
    output = Path(args.out)
    check_outputs(files=[output / 'values_fig4C_four_networks.csv'], inputs=[args.catalog])
    validate_paths([output], [args.catalog])
    catalog = json.loads(Path(args.catalog).read_text())
    sources = [item[key] for network in catalog['networks'].values() for item in network['layers']
               for key in ('source', 'meta') if item.get(key)]
    validate_paths([output], sources)
    import torch
    torch.set_num_threads(1)
    from network_sources import load_network
    output.mkdir(parents=True, exist_ok=True)
    n_layers = 0
    for name in NETWORKS:
        ctx = load_network(name, catalog['networks'][name])
        validate_paths([output], [ctx['mask_root'], ctx['checkpoint']])
        source_record = output / 'sources' / (name + '__shared.json')
        inputs = [args.catalog, *sources, ctx['mask_root'], ctx['checkpoint']]
        validate_paths([source_record], inputs)
        if not source_record.exists():
            atomic_json(source_record,
                    dict(network=name,categories=ctx['categories'],
                         checkpoint=str(ctx['checkpoint']),checkpoint_sha256=ctx['checkpoint_sha256'],
                         masks=[dict(category=int(cat),path=str(path),sha256=digest) for cat,path,digest in
                                zip(ctx['categories'],ctx['mask_paths'],ctx['mask_sha256'])]))
        want = catalog['n_categories'][name]
        if len(ctx['categories']) != want:
            raise ValueError('%s: %d categories, the catalog names %d' % (name, len(ctx['categories']), want))
        results = []
        for item in ctx['layers']:
            target = output / 'layers' / (name+'__'+item['layer']+'.json')
            raw_path = output / 'pair_totals' / (name + '__' + item['layer'] + '.npz')
            validate_paths([target, raw_path], inputs)
            if target.exists():
                result = json.loads(target.read_text())
            else:
                result = layer_task(ctx, item, output)
                atomic_json(target, result)
            results.append(result)
            n_layers += 1
            print(json.dumps(dict(network=name, layer=item['layer'], values=result['values'])), flush=True)
        kept = sum(result['kept']['retained_count_total'] for result in results)
        possible = sum(result['kept']['route_count_total'] for result in results)
        check_printed(100. * kept / possible, TABLE1_KEPT_PERCENT[name])
        del ctx
    assembly = assemble(output)
    if n_layers != N_LAYERS or assembly['rows'] != N_LAYERS:
        raise ValueError('%d layers in the catalog and %d rows written; the four networks have %d pruned layers' %
                         (n_layers, assembly['rows'], N_LAYERS))
    print(json.dumps(assembly), flush=True)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
        parser.add_argument('--mode', choices=['catalog','full'], required=True)
        parser.add_argument('--workspace', default=str(Path(__file__).resolve().parent.parent),
                            help='the folder holding u_xnet.csv and xnet_sign_density.tsv (catalog mode)')
        parser.add_argument('--catalog', default=str(Path(__file__).with_name('source_catalog.json')),
                            help='the table of sources (written in catalog mode, read in full mode)')
        parser.add_argument('--out', help='output folder (full mode)')
        run(parser.parse_args())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
