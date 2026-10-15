"""Hashes of the Z^2 contract's batches, fixed from the code as of commit
4dc94de (2026-10-15). ``tests/test_contract_z2.py`` fails if any later
change to the generator or loader alters a single byte of what an
old-contract model receives."""
import hashlib

import numpy as np

from mapping.data import Config, load_bank, make_batch
from mapping.loader import eval_set


def digest(batch):
    h = hashlib.sha256()
    for k in sorted(batch):
        if k == 'episodes':
            continue
        v = np.ascontiguousarray(batch[k])
        h.update(k.encode()); h.update(str(v.dtype).encode())
        h.update(str(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()[:16]


def cases(path='data/mnist.npz'):
    out = {}
    for dihedral in (False, True):
        bank, _ = load_bank(path, 32, 'train', dihedral)
        for seed in (0, 1):
            c = Config(seed=seed, dihedral=dihedral)
            rng = np.random.default_rng([c.seed, 0])
            out[f'train d{int(dihedral)} s{seed}'] = digest(
                make_batch(rng, c, bank, 8))
        c = Config(dihedral=dihedral)
        ev = eval_set(c, bank, 8, 1, 0)[0]
        out[f'eval d{int(dihedral)}'] = digest(
            {k: v.numpy() for k, v in ev.items()})
    return out


if __name__ == '__main__':
    for k, v in cases().items():
        print(f"    '{k}': '{v}',")
