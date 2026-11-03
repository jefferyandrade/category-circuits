"""Fig. 3A, the frontier comparison: E/I-separate (P) circuits extracted again at a second, higher threshold
(tau = 2.5e-7; the group P-new), where they are smaller, compared category by category with each method's circuits at
the paper's threshold (P, SA-H, SA-G and SF; RP1 where available) on all 1,000 categories. The settings' size target is the
largest mean kept fraction that P-new may have.

The circuits' receipts are read through a manifest that records each receipt's sha256 (--inputs DIR holds
manifest.json and the receipts), and must match the engine, calibration and held-out evaluation of the paper's
circuits. The settings (frontier_settings.json, its sha256 checked) fix the categories, the bootstrap, the methods and
the size target. For each method: the paired differences P-new minus the method in kept weights, corrected accuracy
and loss share, with percentile category bootstrap intervals (10,000 resamples, seed 20261024), marginal and
Bonferroni-corrected for the family of 15 (5 methods x 3 measures), and in hits and correct rejections, with
marginal intervals only. P-new dominates a method when the family intervals show no more kept weights, higher
accuracy and a lower loss share (verdict_family in pairing.csv). Missing categories are reported, not filled in;
inconsistent inputs stop the run (exit status 2).

Units: kept, corrected accuracy, hits and correct rejections are percentage points. The loss share is the held-out
loss increase divided by the training loss span. No test is made per category.

Writes pairing.csv (as in data/frontier/), per_category.csv, summary.json and REPORT.md.
Usage: frontier_analysis.py --inputs DIR --out DIR [--settings frontier_settings.json]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, prepare_dir, publish_file, write_file


SETTINGS = Path(__file__).resolve().with_name('frontier_settings.json')
SETTINGS_SHA256 = '1018cd08d3f63c6bccfc0bb04c5f2de88dd03fe795b51cb10780fdfa686d8297'
_spec = importlib.util.spec_from_file_location('matched_size', Path(__file__).resolve().with_name('matched_size.py'))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
PRIMARY = ('kept', 'accuracy', 'share')
METRICS = ('kept', 'accuracy', 'hits', 'correct_rejections', 'share')
# Receipt fields that must agree: the engine's source (sealed_source_identity), each calibration's procedure, and the
# dense network's calibration.
ENGINE_FIELDS = ('schema', 'seal_kind', 'sealed', 'base_commit', 'source_manifest_sha256')
CALIBRATION_FIELDS = ('calibration_algorithm_id', 'calibration_penalty',
    'calibration_weighting', 'calibration_fit_scope', 'accuracy_threshold')
DENSE_FIELDS = ('calibration_state_id', 'calibration_fit_receipt_hash', 'calibration_affine')
IDENTITIES = M.IDENTITIES


def _digest(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _engine(identity):
    """The engine's identity from a receipt; its source must be identified by a sha256 manifest."""
    result = {key: identity[key] for key in ENGINE_FIELDS}
    if result['sealed'] is not True or result['seal_kind'] != 'sha256_manifest':
        raise ValueError('the engine source is not identified by a sha256 manifest')
    if not result['source_manifest_sha256']:
        raise ValueError('missing engine source manifest hash')
    return result


def _category(value):
    number = M.scalar(value)
    if int(number) != number or not 0 <= number < 1000:
        raise ValueError('invalid category')
    return int(number)


def read_settings(path, expected_sha256):
    """The settings and their sorted categories, after checking the file's sha256 and the bootstrap and methods."""
    blob = Path(path).read_bytes()
    if not expected_sha256 or _digest(blob) != expected_sha256:
        raise ValueError('the settings file is not the expected one: its sha256 differs')
    settings = json.loads(blob)
    categories = [_category(c) for c in settings['categories']]
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError('settings categories must be unique and number at least two')
    if int(settings.get('category_count', len(categories))) != len(categories):
        raise ValueError('settings category count differs')
    bootstrap = settings['bootstrap']
    if (bootstrap['n_resamples'], bootstrap['seed'], bootstrap['family_size']) != (10000, 20261024, 15):
        raise ValueError('settings bootstrap differs from the analysis procedure (10,000 resamples, seed 20261024, family of 15)')
    required = settings.get('required_arms', ['P', 'SA-H', 'SA-G', 'SF'])
    optional = settings.get('optional_arms', ['RP1'])
    if required != ['P', 'SA-H', 'SA-G', 'SF'] or optional != ['RP1']:
        raise ValueError('settings method family differs')
    target = M.scalar(settings['size_target_fraction'])
    if not 0 <= target <= 1:
        raise ValueError('invalid size target in the settings')
    return settings, sorted(categories)


def _protocol(receipt, category):
    """The receipt's identity and protocol fields (model, data, calibration procedure, held-out design), which must
    agree across the circuits of a category; the fits themselves differ from mask to mask."""
    lineage = receipt['terminal_calibration_lineage']
    for key in ('validation_rows_used_for_calibration', 'dense_refit_on_validation', 'final_refit_on_validation'):
        if lineage[key] is not False:
            raise ValueError('held-out rows used in calibration: '+key)
    protocol = {key: receipt[key] for key in IDENTITIES}
    protocol['calibration_deployment_refit_anchor'] = receipt['calibration_deployment_refit_anchor']
    expected_counts = {'on_target': 50}
    expected_counts.update({'off_target:%d' % c: 50 for c in range(1000) if c != category})
    for name in ('terminal_dense_bce', 'terminal_final_bce'):
        block = receipt[name]
        count = M.scalar(block['calibration_context_count'])
        if count < 0 or count != int(count) or (name == 'terminal_final_bce' and count == 0):
            raise ValueError('invalid calibration sample count')
        if block['evaluation_context_counts'] != expected_counts:
            raise ValueError('held-out evaluation counts differ for category %d' % category)
        for key in ('calibration_and_evaluation_disjoint', 'calibration_fit_preexisted_validation'):
            if block[key] is not True:
                raise ValueError('calibration/evaluation protocol differs: '+key)
        protocol[name] = {key: block[key] for key in CALIBRATION_FIELDS}
        protocol[name]['context_design_hash'] = block['context_design_hash']
        if block['context_design_hash'] != receipt['final_validation_context_design_hash']:
            raise ValueError('held-out context design identity differs')
    protocol['dense_calibration'] = {key: receipt['terminal_dense_bce'][key] for key in DENSE_FIELDS}
    for name in ('optimization_data_plane', 'evaluation_data_plane'):
        protocol[name] = {key: receipt[name][key] for key in ('axis_hash', 'preprocessing_hash')}
    return protocol


def _normalize(receipt):
    row = M.normalize(receipt)
    # matched_size stores kept as a fraction; convert kept alone to percentage points.
    row['kept_fraction'] = row['kept']
    row['kept'] *= 100.
    for key in ('route_count', 'retained_route_count'):
        value = M.scalar(receipt[key])
        if value < 0 or int(value) != value:
            raise ValueError('invalid retained weight count')
    if receipt['route_count'] <= 0 or receipt['retained_route_count'] > receipt['route_count']:
        raise ValueError('invalid retained weight count')
    # matched_size.normalize compares int(query_id); a non-integer such as 2.5 must not pass as 2.
    if _category(receipt['query_id']) != row['category']:
        raise ValueError('category/query mismatch')
    operating = receipt['terminal_final_bce']['terminal_statistics_v1']['probability_0_5']
    tpr, fpr = M.scalar(operating['tpr']), M.scalar(operating['fpr'])
    if not (0 <= tpr <= 1 and 0 <= fpr <= 1):
        raise ValueError('invalid held-out hit/rejection rates')
    if not math.isclose(operating['calibrated_logit_cut'], 0., abs_tol=1e-12):
        raise ValueError('changed held-out accuracy threshold')
    if not math.isclose(row['accuracy'], 100*(tpr-fpr), rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError('held-out corrected accuracy disagrees with hits and rejections')
    row['hits'], row['correct_rejections'] = 100*tpr, 100*(1-fpr)
    for name in ('dense', 'final'):
        row[name+'_calibration_context_count'] = receipt['terminal_'+name+'_bce']['calibration_context_count']
    return row


def load_inputs(root, settings, categories):
    """Each group's receipts, by category, checked against the manifest's hashes, the settings and each other."""
    root = Path(root).resolve()
    manifest = _json(root/'manifest.json')
    manifest_records = manifest['records'] if isinstance(manifest, dict) else manifest
    groups = {name: {} for name in [settings.get('new_group', 'P-new')] +
        settings.get('required_arms', ['P', 'SA-H', 'SA-G', 'SF']) + settings.get('optional_arms', ['RP1'])}
    expected_engine = _engine(settings['provenance']['engine_identity'])
    seen = set()
    for record in manifest_records:
        path = (root/record['local']).resolve()
        if root not in path.parents:
            raise ValueError('receipt path lies outside inputs directory')
        blob = path.read_bytes()
        if _digest(blob) != record['sha256']:
            raise ValueError('receipt source hash differs: '+record['local'])
        if record.get('category') is None:
            continue
        category = _category(record['category'])
        group = record['group']
        if group not in groups or category not in categories:
            raise ValueError('unexpected group or category in manifest')
        if (group, category) in seen:
            raise ValueError('duplicate category in '+group)
        seen.add((group, category))
        receipt = json.loads(blob)
        row = _normalize(receipt)
        if row['category'] != category:
            raise ValueError('manifest category mismatch')
        if _engine(receipt['sealed_source_identity']) != expected_engine:
            raise ValueError('engine source manifest differs from the engine of the paper\'s circuits')
        heldout = settings['provenance'].get('heldout', {})
        # The held-out design depends on the category: only the model and the validation images are checked against
        # the settings here, and each circuit's design is checked against its category's P circuit below.
        for key in ('model_state_hash', 'final_validation_context_axis_hash'):
            if key in heldout and receipt[key] != heldout[key]:
                raise ValueError('held-out identity differs from the settings: '+key)
        if group in (settings.get('new_group', 'P-new'), 'P'):
            for key, value in settings['provenance'].get('calibration', {}).items():
                if receipt['effective_settings'].get(key) != value:
                    raise ValueError('calibration or census setting differs from the settings: '+key)
        protocol = _protocol(receipt, category)
        row.update(source=record['source'], source_sha256=record['sha256'],
                   local_receipt=record['local'], group=group, protocol=protocol)
        groups[group][category] = row
    for category in categories:
        rows = [group[category] for group in groups.values() if category in group]
        # Comparisons share the training span, dense calibration and held-out path.
        for row in rows[1:]:
            if row['protocol'] != rows[0]['protocol']:
                raise ValueError('calibration or held-out protocol differs from the P circuit\'s for category %d' % category)
            if not math.isclose(row['training_span'], rows[0]['training_span'], rel_tol=1e-10):
                raise ValueError('training normalization span differs from the P circuit\'s')
    return groups, manifest_records


def compare(new, comparator, categories, n_resamples, seed, incomplete=False):
    """P-new against one method on their paired categories: the differences' intervals, the means and the dominance
    on the marginal and the family intervals."""
    paired = [category for category in categories if category in new and category in comparator]
    output = dict(categories=paired, n_categories=len(paired))
    if len(paired) < 2:
        output.update(verdict_family='insufficient paired categories',
                      verdict_95='insufficient paired categories', mean_only_dominates=False,
                      means={}, contrasts={})
        return output
    arrays = [np.asarray([[group[c][metric] for metric in METRICS] for c in paired], dtype=float)
              for group in (new, comparator)]
    differences = arrays[0]-arrays[1]
    primary_columns = [METRICS.index(metric) for metric in PRIMARY]
    contrasts = M.summarize(differences[:, primary_columns], n_resamples=n_resamples,
                            seed=seed, family_size=15)
    for metric in ('hits', 'correct_rejections'):
        contrasts[metric] = M.block_bootstrap(differences[:, METRICS.index(metric)],
            n_resamples=n_resamples, seed=seed)
    means = {name: dict(zip(METRICS, map(float, array.mean(axis=0))))
             for name, array in zip(('new', 'comparator'), arrays)}
    dominance_95, dominance_family = M.dominance(contrasts), M.dominance(contrasts, 'family_ci_')
    if incomplete:
        dominance_95 = dominance_family = 'incomplete'
    output.update(contrasts=contrasts, means=means, verdict_95=dominance_95,
        verdict_family=dominance_family, mean_only_dominates=(
            contrasts['kept']['estimate'] <= 0 and contrasts['accuracy']['estimate'] > 0
            and contrasts['share']['estimate'] < 0))
    return output


def _write_csv(path, rows, fields):
    with write_file(path, newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _interval(item, family=False):
    prefix = 'family_ci_' if family else 'ci_'
    return '%.6g [%.6g, %.6g]' % (item['estimate'], item[prefix+'low'], item[prefix+'high'])


def write_outputs(out, summary, groups):
    """summary.json, per_category.csv, pairing.csv and REPORT.md."""
    out = Path(out)
    prepare_dir(out)
    publish_file(json.dumps(summary, indent=2, allow_nan=False)+'\n', out/'summary.json')
    fields = ['group', 'category', 'kept_fraction', 'kept', 'accuracy', 'hits',
              'correct_rejections', 'share', 'training_span', 'heldout_dense_bce',
              'heldout_final_bce', 'heldout_delta_bce', 'dense_calibration_context_count',
              'final_calibration_context_count', 'source', 'source_sha256', 'local_receipt']
    rows = [{field: row[field] for field in fields} for group in groups.values()
            for _, row in sorted(group.items())]
    _write_csv(out/'per_category.csv', rows, fields)
    table = []
    for arm, pair in summary['pairs'].items():
        for metric in METRICS:
            if metric not in pair['contrasts']:
                continue
            item = pair['contrasts'][metric]
            row = dict(arm=arm, metric=metric, n_categories=pair['n_categories'],
                new_mean=pair['means']['new'][metric], comparator_mean=pair['means']['comparator'][metric],
                estimate=item['estimate'], ci_low=item['ci_low'], ci_high=item['ci_high'],
                family_ci_low=item.get('family_ci_low', ''), family_ci_high=item.get('family_ci_high', ''),
                verdict_family=pair['verdict_family'])
            table.append(row)
    _write_csv(out/'pairing.csv', table, ['arm', 'metric', 'n_categories', 'new_mean',
        'comparator_mean', 'estimate', 'ci_low', 'ci_high', 'family_ci_low', 'family_ci_high', 'verdict_family'])
    reference = summary['pairs'].get('P', {})
    point_means = reference.get('means', {}).get('new', {})
    point_cells = ['%.6g' % point_means[metric] if metric in point_means else 'unavailable'
                   for metric in METRICS]
    text = ['# Frontier comparison', '',
        'Complete pairing of the main methods: %s. The engine source, calibration procedure and held-out protocol of the paper\'s circuits match on all supplied receipts.' % summary['complete'],
        'Missing categories are recorded without imputation. RP1 uses its available paired categories.', '',
        'The P-new means below are over the categories paired with P at 1.5e-7: %d of the %d categories.' %
        (reference.get('n_categories', 0), len(summary['categories'])), '',
        '| Paired categories | Kept (%) | Corrected accuracy (points) | Hits (%) | Correct rejections (%) | Loss-window share |',
        '|---:|---:|---:|---:|---:|---:|',
        '| %d | %s | %s | %s | %s | %s |' %
        (reference.get('n_categories', 0), point_cells[0], point_cells[1], point_cells[2], point_cells[3], point_cells[4]), '',
        'Differences are P-new minus the method. Kept fraction is in percentage points; corrected accuracy, hits and correct rejections are in points. Loss-window share is dimensionless.',
        'Intervals are percentile category bootstrap intervals, 10,000 resamples, seed 20261024. The intervals in the first table below are Bonferroni-corrected for the family of 15. Their coverage is nominal and the endpoints are Monte Carlo approximations.',
        'Intervals describe category variation given the threshold and the categories. They do not include uncertainty from the choice of threshold, validation-image resampling or calibration fitting.', '',
        '| Method | Paired categories | Kept difference [simultaneous interval] | Corrected accuracy difference [simultaneous interval] | Loss-window share difference [simultaneous interval] | Dominance (simultaneous intervals) | Dominance of the means |',
        '|---|---:|---:|---:|---:|---|---|']
    for arm, pair in summary['pairs'].items():
        cells = [_interval(pair['contrasts'][metric], True) if pair['contrasts'] else 'unavailable'
                 for metric in PRIMARY]
        text.append('| %s | %d | %s | %s | %s | %s | %s |' %
            (arm, pair['n_categories'], cells[0], cells[1], cells[2], pair['verdict_family'], pair['mean_only_dominates']))
    text.extend(['',
        '| Method | Paired categories | Hits difference [marginal 95% interval] | Correct rejections difference [marginal 95% interval] |',
        '|---|---:|---:|---:|'])
    for arm, pair in summary['pairs'].items():
        cells = [_interval(pair['contrasts'][metric]) if metric in pair['contrasts'] else 'unavailable'
                 for metric in ('hits', 'correct_rejections')]
        text.append('| %s | %d | %s | %s |' % (arm, pair['n_categories'], cells[0], cells[1]))
    text.extend(['', 'The paired table in pairing.csv includes marginal 95% intervals for all five metrics and simultaneous primary intervals. No category-specific significance claims are made.', '',
        'The P-new mean kept fraction meets the size target: %s. Missing categories: `%s`.' %
        (summary['size_target_met'], json.dumps(summary['missing_categories'], sort_keys=True))])
    publish_file('\n'.join(text)+'\n', out/'REPORT.md')


def run_analysis(inputs, settings_path, out, settings_sha256):
    """Read and check the inputs, compare P-new with every method and write the outputs; returns the summary."""
    check_outputs(dirs=[out], inputs=[inputs, settings_path])
    try:
        settings, categories = read_settings(settings_path, settings_sha256)
        groups, records = load_inputs(inputs, settings, categories)
    except (KeyError, TypeError) as error:
        raise ValueError('missing or malformed receipt or settings field: %s' % error) from error
    new_group = settings.get('new_group', 'P-new')
    required = settings.get('required_arms', ['P', 'SA-H', 'SA-G', 'SF'])
    optional = settings.get('optional_arms', ['RP1'])
    missing = {name: sorted(set(categories)-set(group)) for name, group in groups.items()}
    complete = all(not missing[name] for name in [new_group]+required)
    pairs = {arm: compare(groups[new_group], groups[arm], categories, 10000, 20261024,
                 incomplete=bool(missing[new_group] or (arm in required and missing[arm])))
             for arm in required+optional}
    observed_new_mean = (float(np.mean([row['kept_fraction'] for row in groups[new_group].values()]))
                         if groups[new_group] else None)
    summary = dict(complete=complete, categories=categories,
        new_group=new_group, n_resamples=10000, seed=20261024, family_size=15,
        family_description='5 comparator arms x 3 primary metrics',
        primary_metrics=list(PRIMARY), supplementary_metrics=['hits', 'correct_rejections'],
        heldout_design_rule='one fixed model and validation axis; category-specific design matched to P',
        calibration_sample_count_rule='record mask-specific counts; compare algorithm, weighting, fit scope and fixed stopping-rule settings',
        units=dict(kept='percentage points', accuracy='corrected accuracy points',
                   hits='percentage points', correct_rejections='percentage points', share='fraction of training loss span'),
        settings_sha256=settings_sha256, source_manifest_sha256=_digest((Path(inputs)/'manifest.json').read_bytes()),
        missing_categories=missing, pairs=pairs, size_target_fraction=settings['size_target_fraction'],
        observed_new_mean_kept_fraction=observed_new_mean,
        size_target_met=(observed_new_mean is not None and not missing[new_group]
                         and observed_new_mean <= settings['size_target_fraction']),
        source_record_count=len(records), engine_identity=_engine(settings['provenance']['engine_identity']))
    write_outputs(out, summary, groups)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', type=Path, required=True, help='folder holding manifest.json and the receipts')
    parser.add_argument('--settings', type=Path, default=SETTINGS,
                        help='the analysis settings (default: the paper\'s, frontier_settings.json)')
    parser.add_argument('--settings-sha256', default=SETTINGS_SHA256, help='the expected sha256 of the settings file')
    parser.add_argument('--out', type=Path, required=True, help='output folder')
    args = parser.parse_args()
    try:
        summary = run_analysis(args.inputs, args.settings, args.out, args.settings_sha256)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print('Could not compute frontier analysis: '+str(error), file=sys.stderr)
        return 2
    print('Computed %d method comparisons; complete pairing: %s' % (len(summary['pairs']), summary['complete']))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
