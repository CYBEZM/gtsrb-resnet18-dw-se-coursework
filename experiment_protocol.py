"""Resolve historical evaluations without rewriting their original provenance."""
from pathlib import Path

from src.utils import read_json, sha256

FIXED_FIELDS = ('manifest_sha256', 'configs', 'corruption_protocol',
                'source_sha256', 'selection', 'evaluation')


def evaluation_protocol_path(active_path, seed):
    active_path = Path(active_path)
    active = read_json(active_path)
    if seed not in active['seeds']:
        raise ValueError(f'Seed {seed} is outside the active experiment plan')
    reference = active.get('inherited_evaluations', {}).get(str(seed))
    if not reference:
        return active_path
    parent_path = Path(reference['path'])
    if sha256(parent_path) != reference['sha256']:
        raise ValueError('Historical evaluation protocol hash mismatch')
    parent = read_json(parent_path)
    if seed not in parent['seeds']:
        raise ValueError('Historical evaluation protocol does not include this seed')
    for field in FIXED_FIELDS:
        if parent[field] != active[field]:
            raise ValueError(f'Historical protocol differs in {field}')
    return parent_path
