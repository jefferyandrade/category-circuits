"""The inputs of ``run_four_networks.py``: each network's layer layout, circuit masks, checkpoint weights and shared
categories, read and checked against every circuit's receipt.

``load_network`` resolves, for one network, the circuit library's folder and the P circuits' folder that its overlap
run (``analysis/stats/overlap_nulls.py``) recorded in the ``meta.json`` beside each layer's overlap summary (AlexNet's
come from its census's command line and summary), the torchvision checkpoint and the shared categories. Every mask is
checked against its receipt (route-axis hash, route count, sha256, zero padding, kept count per layer).
``iter_input_chunks`` yields the masks of a batch of units as (category, unit, input) booleans, a convolution's inputs
being its input channels (kept if any of their kernel weights is kept), with the input weight sizes that define the
deciles: |w| for fully connected weights, a kernel's summed |w| for convolutions. At the end of a layer it checks the
kept counts against the receipts, the overlap run's Gram diagonals and the recorded kept fractions by sign.

``CC_CHECKPOINTS`` names the folder of AlexNet's checkpoint (the other networks' checkpoints are named in their
overlap runs' ``meta.json``). The layer layouts come from ``analysis/stats/circuit_masks.py`` and ``analysis/common``,
the categories from ``taxonomy/inputs/CATEGORY_BLOCKS_cap50.tsv``.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STATS = REPO / 'analysis/stats'
COMMON = REPO / 'analysis/common'
CHECKPOINTS = Path(os.environ.get('CC_CHECKPOINTS',
                                  Path(os.environ.get('TORCH_HOME', Path.home() / '.cache/torch')) / 'hub/checkpoints'))


def sha256_file(path):
    """sha256 of a file."""
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_route_rows(path, offset, start, stop, row_size):
    """Read contiguous rows from the big-endian packed route axis."""
    first, final = int(offset) + int(start)*int(row_size), int(offset) + int(stop)*int(row_size)
    byte_first, bit_first = divmod(first, 8)
    byte_stop = (final+7)//8
    with Path(path).open('rb') as handle:
        handle.seek(byte_first)
        raw = handle.read(byte_stop-byte_first)
    if len(raw) != byte_stop-byte_first:
        raise ValueError('truncated packed mask: '+str(path))
    bits = np.unpackbits(np.frombuffer(raw, np.uint8))
    return bits[bit_first:bit_first+final-first].reshape(stop-start, row_size).astype(bool)


def input_scores(weights):
    """Input weight sizes per output unit: |w| for fully connected weights, a kernel's summed |w| for convolutions."""
    weights = np.asarray(weights)
    scores = np.abs(weights)
    if weights.ndim == 4:
        scores = scores.sum(axis=(2,3))
    elif weights.ndim != 2:
        raise ValueError('expected an FC or convolutional weight array')
    return scores


def _imports():
    """The layer-layout module and the category populations, imported from this repository."""
    for path in [STATS, COMMON]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module('circuit_masks'), importlib.import_module('populations')


def _arg(argv, name):
    return argv[argv.index(name)+1]


def _gram_info(path):
    """Categories and kept counts (the diagonal) of an overlap Gram file."""
    with np.load(path) as z:
        cats = [int(c) for c in z['categories']]
        if len(cats) != len(set(cats)):
            raise ValueError('duplicate category ids in '+str(path))
        key = 'I_all' if 'I_all' in z.files else 'I'
        counts = np.diag(z[key])[:len(cats)].astype(np.int64)
    return cats, dict(zip(cats, counts))


def _bn_module(module):
    """The batch-norm module that follows a ResNet convolution, if any."""
    if module == 'conv1':
        return 'bn1'
    if '.downsample.0' in module:
        return module.replace('.downsample.0','.downsample.1')
    head, sep, tail = module.rpartition('.')
    if tail.startswith('conv') and tail[4:].isdigit():
        return head+sep+'bn'+tail[4:]
    return None


def load_network(network, catalog_network):
    """Resolve exactly the source-named layouts, masks, weights and categories of one network."""
    import torch
    RT, populations = _imports()
    blocks = populations.category_blocks()
    layout = RT.tree_layout(network)
    wanted = list(catalog_network['layers'])
    layers = []
    shared = list(map(int, populations.shared100(blocks)))
    categories = None
    roots, arms, checkpoints = set(), set(), set()
    for record in wanted:
        item = dict(record)
        if network == 'alexnet':
            meta = json.loads(Path(item['source']).read_text())
            argv = meta['argv']
            root, arm = _arg(argv,'--masks-root'), meta['P_arm']
            checkpoint = str(CHECKPOINTS/RT.CHECKPOINT_FILE[network])
            cats = shared
            gram, expected = None, None
            lid = item['layer_id']
        else:
            meta = json.loads(Path(item['meta']).read_text())
            root, arm, checkpoint = meta['results'], meta['arm_p'], meta['checkpoint']
            gram = Path(meta['gram'])
            cats, expected = _gram_info(gram)
            lid = meta['layer_id']
            if not set(cats).issubset(set(shared)) or len(cats) != (99 if network=='resnet50v2' else 100):
                raise ValueError('the categories of the overlap Gram file are not the shared categories: '+str(gram))
        if categories is None:
            categories = cats
        elif categories != cats:
            raise ValueError('layer category order differs inside '+network)
        spec = next(s for s in layout.layers if s.layer_id == lid)
        item.update(spec=spec, meta=meta, gram=str(gram) if gram else None, expected_counts=expected)
        layers.append(item)
        roots.add(str(root)); arms.add(str(arm)); checkpoints.add(str(checkpoint))
    if len(roots)!=1 or len(arms)!=1 or len(checkpoints)!=1:
        raise ValueError('network sources disagree about mask root, arm or checkpoint')
    order = {spec.layer_id:i for i,spec in enumerate(layout.layers)}
    layers.sort(key=lambda item:order[item['spec'].layer_id])
    for i,item in enumerate(layers):
        item['layer_index'] = i+1
    checkpoint = next(iter(checkpoints))
    digest = sha256_file(checkpoint)
    prefix = RT.CHECKPOINT_FILE[network].rsplit('-',1)[-1].split('.')[0]
    if not digest.startswith(prefix):
        raise ValueError('the checkpoint is not the torchvision weights ' + RT.CHECKPOINT_FILE[network])
    for item in layers:
        recorded=item['meta'].get('checkpoint_sha256')
        if recorded and recorded!=digest:
            raise ValueError('checkpoint SHA256 differs from overlap source')
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if isinstance(state,dict) and 'state_dict' in state:
        state=state['state_dict']
    root, arm = Path(next(iter(roots))), next(iter(arms))
    mask_paths, receipt_counts, hashes = [], [], []
    packed_bytes = (int(layout.route_count)+7)//8
    for c in categories:
        directory = root/(network+'_tap_query'+str(c))/arm
        path = directory/'final_mask.packbits'
        receipt = json.loads((directory/'receipt.json').read_text())
        if receipt.get('route_axis_hash') != layout.route_axis_hash or receipt.get('route_count') != layout.route_count:
            raise ValueError('receipt route axis differs from layout: '+str(directory))
        if path.stat().st_size != packed_bytes:
            raise ValueError('packed length differs from layout: '+str(path))
        digest_mask = sha256_file(path)
        if digest_mask != RT.product_sha(receipt,'final_mask.packbits'):
            raise ValueError('mask SHA256 differs from receipt: '+str(path))
        padding = packed_bytes*8-int(layout.route_count)
        if padding:
            with path.open('rb') as f:
                f.seek(-1,2)
                if f.read(1)[0] & ((1<<padding)-1):
                    raise ValueError('nonzero mask padding: '+str(path))
        counts = RT.receipt_expectations(receipt,layout)['retained']
        if sum(counts.values()) != receipt.get('retained_route_count'):
            raise ValueError('receipt layer counts differ from total retained count')
        mask_paths.append(path); receipt_counts.append(counts); hashes.append(digest_mask)
    return dict(network=network,layout=layout,state=state,
                categories=categories,layers=layers,mask_root=root,arm=arm,
                checkpoint=checkpoint,checkpoint_sha256=digest,mask_paths=mask_paths,
                mask_sha256=hashes,receipt_counts=receipt_counts,
                densities=catalog_network.get('densities',{}))


def iter_input_chunks(ctx, item, categories, unit_chunk):
    """Yield (category, unit, input) masks and input weight sizes per batch of units; at the end, check the layer's
    kept counts and record them in ``item['kept']``."""
    if list(map(int,categories)) != ctx['categories']:
        raise ValueError('chunk category identity/order differs from context')
    spec = item['spec']
    weight = ctx['state'][spec.module_name+'.weight'].detach().cpu().numpy()
    if tuple(weight.shape) != tuple(spec.shape):
        raise ValueError('checkpoint shape differs from layout: '+spec.layer_id)
    q = len(categories)
    kept = np.zeros(q,np.int64)
    positive_kept = negative_kept = 0
    bn = _bn_module(spec.module_name) if ctx['network'].startswith('resnet') else None
    if bn and bn+'.weight' in ctx['state']:
        gamma = ctx['state'][bn+'.weight'].detach().cpu().numpy()
        if np.any(gamma==0):
            raise ValueError('a zero batch-norm scale in %s: the deciles of the unfolded weights would differ from '
                             'those of the folded ones' % bn)
    elif ctx['network'].startswith('resnet') and weight.ndim==4:
        raise ValueError('cannot identify batch normalization for '+spec.module_name)
    # Signs before batch-norm folding, as in the recorded kept fractions by sign. Weight-size deciles are the
    # same after folding, which multiplies each output row by a nonzero constant.
    effective_negative=weight<0
    negative_total=int(np.count_nonzero(effective_negative))
    positive_total=int(weight.size-negative_total)
    row_size = int(np.prod(weight.shape[1:]))
    for start in range(0,int(weight.shape[0]),int(unit_chunk)):
        stop=min(start+int(unit_chunk),int(weight.shape[0]))
        rows=np.empty((q,stop-start,row_size),bool)
        neg = effective_negative[start:stop].reshape(stop-start,row_size)
        for c,path in enumerate(ctx['mask_paths']):
            rows[c]=read_route_rows(path,spec.offset,start,stop,row_size)
            kept[c]+=int(rows[c].sum())
            negative_kept+=int(np.count_nonzero(rows[c]&neg))
            positive_kept+=int(np.count_nonzero(rows[c]&~neg))
        if weight.ndim==4:
            masks=rows.reshape(q,stop-start,weight.shape[1],-1).any(axis=-1)
        else:
            masks=rows
        yield masks,input_scores(weight[start:stop])
    expected=np.asarray([counts[spec.layer_id] for counts in ctx['receipt_counts']],dtype=np.int64)
    if not np.array_equal(kept,expected):
        raise ValueError('kept counts of %s differ from the circuits\' receipts' % item['layer'])
    gram=item['expected_counts']
    if gram is not None and not np.array_equal(kept,[gram[int(c)] for c in categories]):
        raise ValueError('kept counts of %s differ from the overlap run\'s Gram file' % item['layer'])
    density_positive=float(positive_kept/(q*positive_total)) if positive_total else None
    density_negative=float(negative_kept/(q*negative_total)) if negative_total else None
    reference=ctx['densities'].get(item['layer'])
    if reference:
        for label,value in [('dens_pos',density_positive),('dens_neg',density_negative)]:
            printed=reference[label]
            places=len(printed.partition('.')[2])
            if value is None or ('%.*f'%(places,value))!=printed:
                raise ValueError('%s: kept fraction %s = %r differs from the recorded %s' %
                                 (item['layer'], label, value, printed))
    item['kept']=dict(retained_count_total=int(kept.sum()),route_count_total=int(q*spec.size),
                      mean_kept_fraction=float(kept.sum()/(q*spec.size)),
                      density_positive=density_positive,density_negative=density_negative)
