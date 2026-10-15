"""Fetch MNIST once and store it as data/mnist.npz.

    python scripts/get_mnist.py

Uses the ``ylecun/mnist`` dataset on the Hugging Face hub through the
``datasets`` library. The brightness cutoff is NOT applied here; the
loader applies it at load time from the config, so the stored file is
the raw data.
"""
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist.npz')


def main():
    if os.path.exists(OUT):
        print(f'{OUT} exists')
        return
    import datasets
    ds = datasets.load_dataset('ylecun/mnist')
    out = {}
    for split in ('train', 'test'):
        d = ds[split].with_format('numpy')
        out[f'{split}_images'] = np.stack(
            [np.asarray(x) for x in d['image']]).astype(np.uint8)
        out[f'{split}_labels'] = np.asarray(d['label']).astype(np.int64)
        print(split, out[f'{split}_images'].shape)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
