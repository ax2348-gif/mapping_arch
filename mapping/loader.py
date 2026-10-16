"""Torch-side loading: the episode stream as a DataLoader, and a fixed
evaluation set.

Training is a stream: batch k is a pure function of (config, seed, k),
nothing is stored. Evaluation is a fixed set of batches drawn from the
bank's *test* split, so no evaluation tile instance was ever seen in
training; it is generated once per run from its own seed.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from mapping.data import (EpisodeStream, check_bank, check_texture,
                          load_bank, make_batch)

TRAIN_SPLIT, EVAL_SPLIT = 'train', 'test'
EVAL_SEED_TAG = 7919   # mixed into the eval seed so it never equals a train seed


class StreamDataset(IterableDataset):
    """Infinite stream; worker j of n takes batch indices j, j + n, ..."""

    def __init__(self, config, bank, batch, start=0):
        self.stream = EpisodeStream(config, bank, batch)
        self.start = start

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (0, 1) if info is None else (info.id,
                                                       info.num_workers)
        for batch in self.stream.batches(worker, workers, self.start):
            batch.pop('episodes')
            yield {k: torch.from_numpy(v) for k, v in batch.items()}


def train_loader(config, bank, batch, workers, start=0):
    """``start`` is the number of batches a resumed run has consumed; the
    stream continues from there with the same multiset of batches as an
    uninterrupted run would have produced."""
    ds = StreamDataset(config, bank, batch, start)
    return DataLoader(ds, batch_size=None, num_workers=workers,
                      persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None,
                      pin_memory=True)


def eval_set(config, bank, batch, batches, seed):
    """``batches`` fixed batches as a list of tensor dicts, on the CPU."""
    rng = np.random.default_rng([seed, EVAL_SEED_TAG])
    out = []
    for _ in range(batches):
        b = make_batch(rng, config, bank, batch)
        b.pop('episodes')
        out.append({k: torch.from_numpy(v) for k, v in b.items()})
    return out


def load_banks(path, config, texture=None):
    """``(train_bank, eval_bank)`` from the bank's two splits, with the
    config's cutoff. Fails here, before any worker starts, if the bank's
    tile side is not ``config.tile`` (``mapping.data.check_bank``) or,
    given the ``texture`` name, if the file's identity tag disagrees with
    it (``mapping.data.check_texture``)."""
    if texture is not None:
        check_texture(path, texture)
    train = load_bank(path, config.cutoff, TRAIN_SPLIT, config.dihedral)[0]
    test = load_bank(path, config.cutoff, EVAL_SPLIT, config.dihedral)[0]
    for bank in (train, test):
        try:
            check_bank(bank, config)
        except ValueError as e:
            raise ValueError(f'{path}: {e}') from None
    return train, test
