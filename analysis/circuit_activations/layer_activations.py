"""Circuit activations and category prototypes at every pruned layer, from every training image (Fig. 7D).

The inputs are the sums that a GPU pass over every training image saved, one folder per image shard. In that pass,
every image went once through the dense network. At each layer's site (the layer's output after its ReLU, averaged
over space for convolutions; for a residual block's last convolution, the block output after the skip path is added),
every circuit whose mask covers the site was applied to the site's dense input, with only that layer's (or block's)
masks, and its responses were summed. Each image of category k (n_k images, K categories) carries weight 1/(n_k K);
the even- and odd-numbered images of each category (halves A and B, counted from zero in the image bank's order) use
their own within-category denominators. A circuit's activation at a site is therefore its mean response over every
training image, with every category weighted equally. The prototypes (the dense network's category means at the same
sites) come from a separate dense pass, one file per site; the dense category means that the shards also hold only
check them.

Modes:
  merge          adds the shards' sums, checks their dense category means against the prototype files and writes each
                 circuit's activations and the prototypes, per population
  aggregate      at one site, the tree shares of the circuits' activations and of the prototypes, their difference with
                 a delete-one-category jackknife interval and a paired category-swap permutation test
                 (layer_statistics.analyze_pair)
  aggregate-all  aggregate at every site of the network

The plotted population is the shared 100 categories. The defaults are the paper's settings:
64 shards, 1,000 category swaps, seed 0.
Usage: layer_activations.py merge --network NET --out DIR --prototypes DENSE_DIR
       layer_activations.py aggregate-all --network NET --out DIR --taxonomy taxonomy/inputs/category_taxonomy.csv
where DIR holds run/NET/shards/<shard>/ and DENSE_DIR the dense pass's comprep_dense_<site>.npz files and
confusion.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import io
import sys
from pathlib import Path

for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import numpy as np

import layer_statistics as S

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_file, publish_files, validate_paths, write_file

# Schema labels of the saved files: the GPU pass's (also written into MERGED.json), the merged per-circuit files'
# (checked by aggregate) and the per-layer results' (read by collect_layer_table.py).
SHARD_SCHEMA = 'alldata-nets-comprep-v1'
CIRCUIT_SCHEMA = 'comprep-x-v1'
RESULT_SCHEMA = 'comprep-x-aggregation-v1'
HALVES = ('mean', 'A', 'B')


class SiteSums:
    """A site's float64 sums of circuit responses and the images counted in each category and half."""
    def __init__(self, counts, circuits, features, halves='odd_even'):
        self.counts = np.asarray(counts, dtype=np.int64)
        if (self.counts.ndim != 1 or not len(self.counts) or (self.counts < 2).any()
                or circuits < 1 or features < 1 or halves != 'odd_even'):
            raise ValueError('counts, shape or half definition is invalid')
        self.halves = halves
        self.sums = np.zeros((3, circuits, features), dtype=np.float64)
        self.seen = np.zeros((len(self.counts), 3), dtype=np.int64)

    def finish(self):
        half_a = (self.counts + 1) // 2
        expected = np.stack([self.counts, half_a, self.counts - half_a], axis=1)
        if not np.array_equal(self.seen, expected):
            raise ValueError('the shards do not count every image of every category exactly once')
        return {name: self.sums[i] / len(self.counts) for i, name in enumerate(HALVES)}


def merge_sums(values):
    if not values:
        raise ValueError('no shard sums')
    first = values[0]
    result = SiteSums(first.counts, first.sums.shape[1], first.sums.shape[2], first.halves)
    for value in values:
        if (not np.array_equal(value.counts, first.counts) or value.halves != first.halves
                or value.sums.shape != first.sums.shape):
            raise ValueError('shard counts, halves or feature shape differ')
        result.sums += value.sums
        result.seen += value.seen
    return result


def load_sums(path):
    with np.load(path, allow_pickle=False) as archive:
        sums = np.asarray(archive['sums'])
        if sums.ndim != 3 or not np.isfinite(sums).all():
            raise ValueError('shard sums must be finite three-dimensional arrays')
        value = SiteSums(archive['counts'], sums.shape[1], sums.shape[2], str(archive['halves']))
        seen = np.asarray(archive['seen'])
        if seen.shape != value.seen.shape or seen.dtype.kind not in 'iu' or (seen < 0).any():
            raise ValueError('shard image count shape or dtype differs')
        value.sums[:] = sums
        value.seen[:] = seen
    return value


def validate_shards(records, nshards):
    if len(records) != nshards or sorted(r['shard'] for r in records) != list(range(nshards)):
        raise ValueError('missing or duplicate shard ids')
    for record in records:
        for key in ('schema', 'network', 'identity', 'categories', 'counts', 'circuit_keys', 'nshards',
                    'sites', 'populations', 'circuits', 'batch_size', 'half_definition'):
            if record.get(key) != records[0].get(key):
                raise ValueError('shard ' + key + ' differs')
        if record['nshards'] != nshards:
            raise ValueError('shard partition differs')


def compare_vectors(actual, reference, atol=5e-5, rtol=5e-5):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        return dict(passed=False, reason='shape differs', actual_shape=list(actual.shape), reference_shape=list(reference.shape))
    return dict(passed=bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
                max_abs=float(np.max(np.abs(actual - reference))), atol=atol, rtol=rtol)


def all_passed(checks):
    """'pass' when every comparison agrees, else 'fail': the value MERGED.json records under 'verdict'."""
    return 'pass' if checks and all(value['passed'] for value in checks.values()) else 'fail'


def read_taxonomy(path):
    """The category tree: each category's animate, superordinate and block50 labels."""
    result = {}
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        needed = {'category', 'animate', 'superordinate', 'block50'}
        if not needed.issubset(set(reader.fieldnames or ())):
            raise ValueError('taxonomy must supply category, animate, superordinate, block50')
        for row in reader:
            category = int(row['category'])
            if category in result:
                raise ValueError('taxonomy category ids must be unique')
            labels = {column: row[column].strip() for _, column in S.LEVELS}
            if any(not label or label.lower() in ('nan', 'none') for label in labels.values()):
                raise ValueError('taxonomy has missing labels')
            result[category] = labels
    return result


def _dump(path, value):
    path = Path(path)
    publish_file(json.dumps(value, indent=2, allow_nan=False) + '\n', path)


def read_confusion(prototypes):
    """The dense network's top-1 counts over all 1,000 categories and outputs (confusion.npz of the dense pass)."""
    path = Path(prototypes)
    if path.is_dir():
        path = path / 'confusion.npz'
    with np.load(path, allow_pickle=False) as archive:
        for key in ('top1', 'top1_counts', 'moment_mean'):
            if key in archive:
                result = np.asarray(archive[key])
                break
        else:
            raise ValueError('the confusion archive has no top-1 count matrix')
        if ('stratum_order' not in archive or 'logit_order' not in archive
                or not np.array_equal(archive['stratum_order'], np.arange(1000))
                or not np.array_equal(archive['logit_order'], np.arange(1000))):
            raise ValueError('the confusion archive\'s category or output axis differs from all 1000 categories')
    if result.shape != (1000, 1000) or result.dtype.kind not in 'iu' or (result < 0).any():
        raise ValueError('confusion counts must be the full 1000-way integer matrix')
    return result


def load_prototypes(prototypes, site, categories, counts, features, image_membership,
                    state_hash=None, axis_hash=None):
    """Read the prototypes (the dense network's category means and their halves) at one site."""
    root = Path(prototypes)
    if root.is_file():
        root = root.parent
    path = root / ('comprep_dense_' + site['name'] + '.npz')  # the dense pass's file name for a site
    expected_categories = np.asarray(categories)
    expected_counts = np.asarray(counts)
    with np.load(path, allow_pickle=False) as archive:
        if archive['class_ids'].dtype.kind not in 'iu' or not np.array_equal(archive['class_ids'], expected_categories):
            raise ValueError('prototype file category axis differs')
        if archive['counts'].dtype.kind not in 'iu' or not np.array_equal(archive['counts'], expected_counts):
            raise ValueError('prototype file image counts differ')
        if str(archive['membership_sha256']) != image_membership:
            raise ValueError('prototype file images differ from the shards\'')
        if state_hash is not None and str(archive['state_sha256']) != state_hash:
            raise ValueError('prototype file model weights differ from the shards\'')
        if axis_hash is not None and str(archive['axis_hash']) != axis_hash:
            raise ValueError('prototype file image bank differs from the shards\'')
        if json.loads(str(archive['site_metadata'])) != site:
            raise ValueError('prototype file site definition differs')
        if (archive['A_counts'].dtype.kind not in 'iu' or archive['B_counts'].dtype.kind not in 'iu'
                or not np.array_equal(archive['A_counts'], (expected_counts + 1) // 2)
                or not np.array_equal(archive['B_counts'], expected_counts // 2)):
            raise ValueError('prototype file half counts differ')
        result = {}
        for half in HALVES:
            value = np.asarray(archive[half])
            if value.shape != (len(categories), features) or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError('prototypes must be finite category-by-feature arrays')
            result[half] = value.astype(np.float64)
    weighted_halves = (result['A'] * ((expected_counts + 1) // 2)[:, None]
                       + result['B'] * (expected_counts // 2)[:, None]) / expected_counts[:, None]
    if not np.allclose(result['mean'], weighted_halves, rtol=1e-10, atol=1e-12):
        raise ValueError('prototype file mean differs from its count-weighted halves')
    return result


def merge(args):
    base = args.out.resolve() / 'run' / args.network
    validate_paths([base / 'MERGED.json'], [base / 'shards', args.prototypes])
    if (base / 'MERGED.json').exists():
        print('Kept completed merge: %s' % (base / 'MERGED.json'))
        return
    directories = [base / 'shards' / ('%05d' % i) for i in range(args.nshards)]
    records = [json.loads((d / 'COMPLETE.json').read_text()) for d in directories]
    validate_shards(records, args.nshards)
    first = records[0]
    destinations = [base / 'cache/COMPLETE.json', base / 'MERGED.json']
    destinations += [base / 'cache' / p / 'dense.npz' for p in first['populations']]
    destinations += [base / p / ('%s_%d%s' % (c['arm'], c['category'], suffix))
                     for p, cats in first['populations'].items() for c in first['circuits']
                     if c['category'] in cats and (p == 'shared' or c['arm'] == 'primary')
                     for suffix in ('.npz', '.json')]
    prototype_root = Path(args.prototypes)
    if prototype_root.is_file():
        prototype_root = prototype_root.parent
    validate_paths(destinations, [base / 'shards', prototype_root])
    check_outputs(files=[p for p in destinations if not p.is_file()],
                  inputs=[base / 'shards', prototype_root])
    confusion = read_confusion(args.prototypes)
    if not np.array_equal(confusion.sum(1), first['counts']):
        raise ValueError('the confusion counts and the shards differ in image counts')
    dense_by_population = {p: dict(categories=np.asarray(cats), confusion_counts=confusion[cats])
                           for p, cats in first['populations'].items()}
    circuit_arrays = {(p, c['key']): dict(category=np.asarray(c['category']), categories=np.asarray(cats))
                      for p, cats in first['populations'].items() for c in first['circuits']
                      if c['category'] in cats and (p == 'shared' or c['arm'] == 'primary')}
    dense_checks = {}
    prototype_files = []
    prototype_root = Path(args.prototypes)
    if prototype_root.is_file():
        prototype_root = prototype_root.parent
    for site in first['sites']:
        name = site['name']
        paths = [d / (name + '.npz') for d in directories]
        combined = merge_sums([load_sums(p) for p in paths])
        summaries = combined.finish()
        dense = None
        keys = None
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                if keys is None:
                    keys = archive['circuit_keys'].tolist()
                elif keys != archive['circuit_keys'].tolist():
                    raise ValueError('shard circuit ordering differs at site ' + name)
                value = np.asarray(archive['dense'])
                if value.shape != (3, len(first['categories']), combined.sums.shape[-1]) or not np.isfinite(value).all():
                    raise ValueError('dense site shard shape or values differ')
                dense = value.copy() if dense is None else dense + value
        prototypes = load_prototypes(args.prototypes, site, first['categories'], first['counts'], combined.sums.shape[-1],
                                     first['identity']['all_training_membership_sha256'],
                                     state_hash=first['identity']['state_hash'], axis_hash=first['identity']['axis_hash'])
        prototype_files.append(str(prototype_root / ('comprep_dense_' + name + '.npz')))
        for half_index, half in enumerate(HALVES):
            dense_checks[name + ':' + half] = compare_vectors(dense[half_index], prototypes[half], atol=2e-5, rtol=2e-5)
        for population, cats in first['populations'].items():
            if population == 'all' and not name.startswith('stage4.'):
                continue
            indices = [first['categories'].index(c) for c in cats]
            for half_index, half in enumerate(HALVES):
                dense_by_population[population][name + '__' + half] = prototypes[half][indices]
            for key in keys:
                if (population, key) in circuit_arrays:
                    for half in HALVES:
                        circuit_arrays[(population, key)][name + '__' + half] = summaries[half][keys.index(key)]
    for population, arrays in dense_by_population.items():
        target = base / 'cache' / population
        target.mkdir(parents=True, exist_ok=True)
        if not (target / 'dense.npz').exists():
            with write_file(target / 'dense.npz', 'wb') as stream:
                np.savez(stream, **arrays)
    for (population, key), arrays in circuit_arrays.items():
        circuit = next(c for c in first['circuits'] if c['key'] == key)
        stem = base / population / ('%s_%d' % (circuit['arm'], circuit['category']))
        stem.parent.mkdir(parents=True, exist_ok=True)
        if not stem.with_suffix('.npz').exists():
            with write_file(stem.with_suffix('.npz'), 'wb') as stream:
                np.savez(stream, **arrays)
        if not stem.with_suffix('.json').exists():
            _dump(stem.with_suffix('.json'), dict(schema=CIRCUIT_SCHEMA, network=args.network,
                population=population, category=circuit['category'], arm=circuit['arm'],
                identity=first['identity'], sites=first['sites'], walk=circuit['walk']))
    if not (base / 'cache/COMPLETE.json').exists():
        _dump(base / 'cache/COMPLETE.json', dict(schema=CIRCUIT_SCHEMA, identity=first['identity'],
            populations=first['populations'], sites=first['sites'], image_counts=first['counts'],
            computation_image_categories=first['categories'], half_definition=first['half_definition'],
            image_representation_source='prototype files of the dense pass', image_source_files=prototype_files))
    verdict = all_passed(dense_checks)
    _dump(base / 'MERGED.json', dict(schema=SHARD_SCHEMA, network=args.network, nshards=args.nshards,
        total_images=sum(first['counts']), counts=first['counts'], categories=first['categories'],
        prototypes=str(args.prototypes), image_source_files=prototype_files, dense_checks=dense_checks,
        verdict=verdict))
    print(json.dumps(dict(dense_means_match_prototypes=verdict, network=args.network,
                          total_images=sum(first['counts']))))


def aggregate(args):
    """layer_statistics.analyze_pair at one site, on the merged category means."""
    root = args.out.resolve()
    base = root / 'run' / args.network
    target = args.result.resolve() if args.result else root / 'run/aggregation' / ('%s_%s_%s.json' % (args.network, args.population, args.site))
    validate_paths([target, target.with_suffix('.npz')], [base, args.taxonomy])
    if target.exists():
        print('Kept completed layer: %s' % target)
        return
    check_outputs(files=[target, target.with_suffix('.npz')],
                  inputs=[base, args.taxonomy])
    if json.loads((base / 'MERGED.json').read_text())['verdict'] != 'pass':
        raise ValueError('the shards\' dense category means differ from the prototype files (see MERGED.json)')
    record = json.loads((base / 'cache/COMPLETE.json').read_text())
    categories = record['populations'][args.population]
    site = next(s for s in record['sites'] if s['name'] == args.site)
    with np.load(base / 'cache' / args.population / 'dense.npz', allow_pickle=False) as archive:
        dense = {h: np.asarray(archive[args.site + '__' + h]) for h in HALVES}
        counts = np.asarray(archive['confusion_counts'])
        if archive['categories'].tolist() != categories:
            raise ValueError('dense category order differs')
        for h in HALVES:
            if dense[h].ndim != 2 or dense[h].shape[0] != len(categories) or not np.isfinite(dense[h]).all():
                raise ValueError('dense site values or category shape differ')
    fields = {}
    walks = {}
    sources = []
    for arm in (('primary', 'redraw') if args.population == 'shared' else ('primary',)):
        rows = {h: [] for h in HALVES}
        walks[arm] = {'deep_first_categories': [], 'reconstructed_categories': [], 'orders': []}
        for c in categories:
            stem = base / args.population / ('%s_%d' % (arm, c))
            metadata = json.loads(stem.with_suffix('.json').read_text())
            expected = dict(schema=CIRCUIT_SCHEMA, identity=record['identity'], category=c,
                            network=args.network, population=args.population, arm=arm)
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError('circuit identity/category/population/arm differs')
            with np.load(stem.with_suffix('.npz'), allow_pickle=False) as archive:
                if archive['categories'].tolist() != categories:
                    raise ValueError('circuit category axis differs')
                for h in HALVES:
                    row = np.asarray(archive[args.site + '__' + h])
                    if row.ndim != 1 or not np.isfinite(row).all() or row.shape != dense[h].shape[1:]:
                        raise ValueError('circuit vector shape or values differ')
                    rows[h].append(row)
            walks[arm]['deep_first_categories'].append(c)
            sources.append(str(stem))
        fields[arm] = {h: np.stack(v) for h, v in rows.items()}
    second = fields.get('redraw')
    result = S.analyze_pair(fields['primary']['mean'], dense['mean'], categories,
        read_taxonomy(args.taxonomy), counts, second=second['mean'] if second is not None else None,
        seed=args.seed, permutations=args.permutations)
    if second is None:
        reason = 'the P-redraw circuits cover only the shared categories'
        result['first_minus_second'] = {level: dict(S.undefined(reason), difference=None) for level, _ in S.LEVELS}
    arrays = dict(categories=np.asarray(categories))
    for name, family in result['families'].items():
        similarity = family.pop('pearson_similarity')
        key = 'similarity_' + name
        if similarity is not None:
            arrays[key] = np.asarray(similarity)
        family['pearson_similarity_array'] = key if similarity is not None else None
    arrays['confusion_pair_rate'] = np.asarray(result['confusion'].pop('pair_rate'))
    result['confusion']['pair_rate_array'] = 'confusion_pair_rate'
    archive = io.BytesIO()
    np.savez(archive, **arrays)
    result.update(schema=RESULT_SCHEMA, network=args.network, population=args.population, site=site,
        identity=record['identity'], walks=walks, source_stems=sources,
        second_run=dict(status='formed' if second is not None else 'unavailable_for_population'),
        taxonomy=str(args.taxonomy), similarity_archive=str(target.with_suffix('.npz')),
        all_image_counts=record['image_counts'],
        computation_image_categories=record['computation_image_categories'], half_definition=record['half_definition'],
        image_representation_source=record['image_representation_source'], image_source_files=record['image_source_files'])
    publish_files([(archive.getvalue(), target.with_suffix('.npz')),
                   (json.dumps(result, indent=2, allow_nan=False) + '\n', target)])
    print(json.dumps(dict(result=str(target), n_categories=len(categories))))


def aggregate_all(args):
    record = json.loads((args.out.resolve() / 'run' / args.network / 'cache/COMPLETE.json').read_text())
    for population in record['populations']:
        for site in record['sites']:
            if population == 'all' and not site['name'].startswith('stage4.'):
                continue
            current = argparse.Namespace(**vars(args))
            current.population, current.site, current.result = population, site['name'], None
            aggregate(current)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('merge', 'aggregate', 'aggregate-all'))
    parser.add_argument('--network', required=True, choices=('alexnet', 'vgg11', 'resnet18', 'resnet50v1'))
    parser.add_argument('--out', type=Path, required=True,
                        help='the folder holding run/NETWORK/shards; every output is written below it')
    parser.add_argument('--nshards', type=int, default=64, help='the number of image shards (merge)')
    parser.add_argument('--prototypes', type=Path,
                        help='the dense pass folder, with comprep_dense_<site>.npz and confusion.npz (merge)')
    parser.add_argument('--population', choices=('shared', 'all'), default='shared',
                        help='shared: the shared categories; all: every category with a ResNet-18 circuit, at stage 4')
    parser.add_argument('--site', help='the layer (aggregate)')
    parser.add_argument('--taxonomy', type=Path, help='taxonomy/inputs/category_taxonomy.csv (aggregate)')
    parser.add_argument('--result', type=Path, help='the result JSON (aggregate; default run/aggregation/)')
    parser.add_argument('--permutations', type=int, default=1000, help='paired category swaps')
    parser.add_argument('--seed', type=int, default=0, help='the seed of the category swaps')
    args = parser.parse_args(argv)
    if args.nshards < 1:
        parser.error('--nshards must be positive')
    if args.mode == 'merge' and args.prototypes is None:
        parser.error('merge requires --prototypes')
    if args.mode == 'aggregate' and (args.site is None or args.taxonomy is None):
        parser.error('aggregate requires --site and --taxonomy')
    if args.mode == 'aggregate-all' and args.taxonomy is None:
        parser.error('aggregate-all requires --taxonomy')
    {'merge': merge, 'aggregate': aggregate, 'aggregate-all': aggregate_all}[args.mode](args)


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
