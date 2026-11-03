"""Collect the per-layer results of layer_activations.py into one table (the input of Fig. 7D).

Reads every per-site result JSON that layer_activations.py aggregate wrote in the current folder (its run/aggregation
folder) and prints one TSV row per network, population and site: the tree shares of the circuits' activations (C,
computation), the prototypes (I, image) and the P-redraw circuits' activations (C2), and C minus I (D) with its
jackknife interval and permutation p, at the domain (dom), superordinate (sup) and block50 (b50) levels, followed by
both representations' confusion Spearman correlations (confC, confI). The paper's table is this output, saved as
site_tree_shares.tsv. With a path argument it also writes every value at full precision as JSON; interval signs are
read from the full-precision values, not the rounded TSV.
Usage: cd AGGREGATION && python3 collect_layer_table.py [FULL.json] > site_tree_shares.tsv
"""
import json, glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, write_file
sources = sorted(glob.glob('*.json'))
if len(sys.argv) > 1:
    try:
        check_outputs(files=[sys.argv[1]], inputs=sources)
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
rows = []
full = []
for f in sources:
    d = json.load(open(f))
    # The schema label that layer_activations.py writes into each per-layer result.
    if d.get('schema') != 'comprep-x-aggregation-v1':
        continue
    fam = d['families']; cmi = d['computation_minus_image']
    out = [d['network'], d['population'], d['site']['name'], str(d['n_categories'])]
    rec = {'network': d['network'], 'population': d['population'], 'site': d['site']['name'],
           'n_categories': d['n_categories'], 'reduction': d['site'].get('reduction'),
           'response': d['site'].get('response'), 'source': f, 'levels': {}}
    for lev in ('domain', 'superordinate', 'block50'):
        c = fam['computation']['tree'][lev].get('share_adjusted')
        i = fam['image']['tree'][lev].get('share_adjusted')
        s = fam.get('second', {}).get('tree', {}).get(lev, {}).get('share_adjusted')
        m = cmi[lev]
        jk = m.get('jackknife', {})
        rec['levels'][lev] = {'computation': c, 'image': i, 'second': s, 'difference': m['difference'],
                              'lo': jk.get('ci_low'), 'hi': jk.get('ci_high'),
                              'p_permutation': m.get('permutation', {}).get('p_two_sided')}
        out += ['%.3f' % c if c is not None else 'NA', '%.3f' % i if i is not None else 'NA',
                '%.3f' % s if s is not None else 'NA',
                '%+.3f [%+.3f,%+.3f] p=%.3f' % (m['difference'], jk.get('ci_low', float('nan')), jk.get('ci_high', float('nan')),
                                                 m.get('permutation', {}).get('p_two_sided', float('nan')))]
    out += ['%.3f' % fam['computation']['confusion_spearman']['rho'], '%.3f' % fam['image']['confusion_spearman']['rho']]
    rows.append('\t'.join(out))
    full.append(rec)
print('net\tpop\tsite\tn\tdomC\tdomI\tdomC2\tdomD\tsupC\tsupI\tsupC2\tsupD\tb50C\tb50I\tb50C2\tb50D\tconfC\tconfI')
print('\n'.join(rows))
if len(sys.argv) > 1:
    with write_file(sys.argv[1]) as stream:
        json.dump(full, stream, indent=1)
