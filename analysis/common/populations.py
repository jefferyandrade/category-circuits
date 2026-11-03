"""The categories of the analyses: the 100 shared categories and each category's subgroup of the category tree (a
WordNet block of at most 50 categories; 131 in all), read from taxonomy/inputs/CATEGORY_BLOCKS_cap50.tsv, whose sha256
is checked."""
from __future__ import annotations
import csv
import hashlib
import io
from pathlib import Path
import numpy as np
BLOCKS_TABLE = Path(__file__).resolve().parents[2] / 'taxonomy' / 'inputs' / 'CATEGORY_BLOCKS_cap50.tsv'
BLOCKS_SHA256 = '1c658006699eb3a8c8d2aa57b780edd3290f7d3b4e30402a4934021fccec56be'
BLOCK_MAP = 'cap50'

def category_blocks(path: Path=BLOCKS_TABLE) -> dict:
    """Each category's block and the shared categories: {'maps': {'cap50': {category: block}}, 'shared100': [...]}."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BLOCKS_SHA256:
        raise ValueError(f'{path}: not the category block table of the paper')
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter='\t'))
    return {'maps': {BLOCK_MAP: {str(int(r['class_index'])): r['block_cap50'] for r in rows}},
            'shared100': [int(r['class_index']) for r in rows if r['shared100'] == '1']}

def shared100(blocks: dict | None=None) -> tuple[int, ...]:
    """The 100 shared categories, in ascending order."""
    return tuple(sorted((int(c) for c in (blocks or category_blocks())['shared100'])))

def labels(categories, blocks: dict | None=None) -> np.ndarray:
    """Block labels 0..G-1 (sorted block names) of ``categories`` in the order given."""
    mapping = (blocks or category_blocks())["maps"][BLOCK_MAP]
    names = [mapping[str(int(c))] for c in categories]
    ids = {name: i for i, name in enumerate(sorted(set(names)))}
    return np.array([ids[name] for name in names], dtype=np.int64)
