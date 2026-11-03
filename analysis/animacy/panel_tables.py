"""The plotted values of Fig. 5A-B as tables, each with a column record.

``write_animacy_tables`` reads AlexNet's animacy statistics (``run_alexnet_animacy.py statistics``) and writes the
joint count grids, the participation-specificity plane with its hypergeometric null (evaluating ``plane`` and
``hyp_pmf`` from the plane helper ``specificity_plane.py``), the participation curves and the layer statistics. Every
CSV gets a ``.columns.json`` record naming the inputs and any transform behind each column.
"""
from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_files


def _write(stem, name, frame, sources, population):
    frame = frame.copy()
    # pandas' default float32 text can round the stored arrays.
    # Widen only for decimal serialization so reading the CSV recovers them.
    for column in frame.select_dtypes(include=['float16', 'float32']):
        frame[column] = frame[column].astype(np.float64)
    frame['population_size'] = population['size']
    path = stem.with_name(stem.name + '_' + name + '.csv')
    columns = {}
    for column in frame:
        source = population['source'] if column == 'population_size' else sources.get(column, sources.get('*'))
        if source is None:
            raise ValueError(f'{path.name}: source missing for {column}')
        columns[column] = source
    publish_files([(frame, path), (json.dumps(dict(
        figure=stem.name, population_size=population['size'], columns=columns), indent=2) + '\n',
        path.with_suffix('.columns.json'))])
    return path


def _source(path, transform=None):
    value = dict(file=str(Path(path).resolve()))
    if transform:
        value['transform'] = transform
    return value


def write_animacy_tables(stem, statistics, script):
    """Write the tables ``<stem>_joint_counts.csv``, ``_plane.csv``, ``_participation_curves.csv`` and
    ``_layer_statistics.csv`` from the statistics folder ``statistics``; ``script`` is the plane helper. Returns
    their paths."""
    stem, statistics = Path(stem), Path(statistics)
    names = ['joint_counts', 'plane', 'participation_curves']
    if (statistics / 'semantic_layer_stats_alexnet1000.parquet').exists():
        names.append('layer_statistics')
    outputs = [stem.with_name(stem.name + '_' + name + suffix) for name in names
               for suffix in ('.csv', '.columns.json')]
    check_outputs(files=outputs, inputs=[statistics, script])
    mp = statistics / 'meta_alexnet1000.json'
    gp = statistics / 'domain_grids_alexnet1000.npz'
    cp = statistics / 'semantic_curves_alexnet1000.npz'
    meta, grids, curves = json.loads(mp.read_text()), np.load(gp), np.load(cp)
    population = dict(size=int(meta.get('K', meta.get('n_animate', 0) + meta.get('n_inanimate', 0))),
                      source=_source(mp, 'K, or n_animate + n_inanimate'))
    # Evaluate the plane helper's own functions with this population's group sizes, bins and EPS.
    # Only its function definitions are loaded; nothing else in the file runs.
    from scipy.special import gammaln
    tree = ast.parse(Path(script).read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'hyp_pmf', 'plane'}]
    namespace = dict(np=np, gammaln=gammaln, nA=meta['n_animate'], nI=meta['n_inanimate'],
                     K=meta['n_animate'] + meta['n_inanimate'], EPS=1e-9,
                     NQ=40, NS=40, qb=np.linspace(0, 1, 41), sb=np.linspace(-1, 1, 41))
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(script), 'exec'), namespace)
    joint, planes, bins = [], [], []
    for key in grids.files:
        if not key.endswith('|H'):
            continue
        layer, H = key[:-2], grids[key].astype(float)
        a, b = np.indices(H.shape)
        joint.append(pd.DataFrame(dict(layer=layer, first_count=a.ravel(), second_count=b.ravel(),
                                       retained_route_count=H.ravel())))
        observed, null, marginal = namespace['plane'](H)
        si, qi = np.indices(observed.shape)
        planes.append(pd.DataFrame(dict(layer=layer, participation_bin=qi.ravel(), specificity_bin=si.ravel(),
            q_low=namespace['qb'][qi.ravel()], q_high=namespace['qb'][qi.ravel() + 1],
            s_low=namespace['sb'][si.ravel()], s_high=namespace['sb'][si.ravel() + 1],
            q_center=.5 * (namespace['qb'][qi.ravel()] + namespace['qb'][qi.ravel() + 1]),
            s_center=.5 * (namespace['sb'][si.ravel()] + namespace['sb'][si.ravel() + 1]),
            observed_route_count=observed.ravel(), null_route_count=null.ravel(),
            plotted_route_count=np.where(observed > 0, observed, np.nan).ravel(),
            participation_route_count=marginal[qi.ravel()])))
        edges = curves['qedges']
        row = dict(layer=layer, participation_bin=np.arange(len(edges) - 1), q_low=edges[:-1], q_high=edges[1:])
        for suffix in ('obs_curve', 'null_curve_mean', 'null_curve_sd', 'envelope_bound', 'obs_wsum', 'exceed'):
            curve_key = layer + '|' + suffix
            if curve_key in curves:
                row[suffix] = curves[curve_key]
        bins.append(pd.DataFrame(row))
    sources = {'*': dict(files=[str(gp.resolve()), str(mp.resolve())],
                        transform='plane(H): participation and specificity bins, hypergeometric null'),
               'population_size': _source(mp, 'n_animate + n_inanimate')}
    def write(name, frame, inputs):
        frame = frame.copy()
        frame['first_group'] = meta.get('first_group', 'animate')
        frame['second_group'] = meta.get('second_group', 'inanimate')
        frame['n_first'], frame['n_second'] = meta['n_animate'], meta['n_inanimate']
        labels = {key: _source(mp) for key in ('first_group', 'second_group', 'n_first', 'n_second')}
        return _write(stem, name, frame, dict(inputs, **labels), population)
    paths = [write('joint_counts', pd.concat(joint, ignore_index=True),
                   {'*': _source(gp), 'population_size': sources['population_size']}),
             write('plane', pd.concat(planes, ignore_index=True), sources),
             write('participation_curves', pd.concat(bins, ignore_index=True),
                   {'*': _source(cp), 'population_size': sources['population_size']})]
    stats = statistics / 'semantic_layer_stats_alexnet1000.parquet'
    if stats.exists():
        paths.append(write('layer_statistics', pd.read_parquet(stats),
            {'*': _source(stats), 'population_size': sources['population_size']}))
    return paths
