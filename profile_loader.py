"""Profile episode generation and the torch loader around it.

Reports a per-stage breakdown of one episode, throughput of the numpy
batch path, throughput through a torch DataLoader at several worker
counts, bytes per batch, and the query statistics that decide whether the
task is retrieval or rendering: how many valid query corners an episode
offers and what fraction of sampled queries sit off the step lattice.
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from mapping.data import (Config, EpisodeStream, bank_size, coverage,
                          crop_windows, load_bank, make_batch, make_episode,
                          random_walk, render_map, sample_queries)


class StreamDataset(IterableDataset):
    def __init__(self, config, bank, batch):
        self.stream = EpisodeStream(config, bank, batch)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (0, 1) if info is None else (info.id,
                                                       info.num_workers)
        for batch in self.stream.batches(worker, workers):
            batch.pop('episodes')
            yield {k: torch.from_numpy(v) for k, v in batch.items()}


def stage_breakdown(config, bank, episodes=2000, seed=1):
    """Microseconds per episode in each stage, measured separately."""
    c = config
    rng = np.random.default_rng(seed)
    times = {}

    t0 = time.perf_counter()
    directions, displacement = random_walk(rng, episodes, c.steps, c.step)
    times['walk (batched)'] = time.perf_counter() - t0

    eps, maps, last = [], [], []
    t0 = time.perf_counter()
    for i in range(episodes):
        ep, seen = make_episode(rng, c, bank_size(bank), displacement[i])
        eps.append(ep)
        last.append(seen)
    times['episode: tiles+coverage+queries'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for ep in eps:
        maps.append(render_map(bank, ep.tile_grid, c.tile, ep.tile_sym))
    times['render map'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for ep, image in zip(eps, maps):
        crop_windows(image, ep.positions, c.obs)
        crop_windows(image, ep.queries, c.obs)
    times['crop obs+targets'] = time.perf_counter() - t0

    # the coverage and query sampling alone, to see what the episode
    # stage is made of
    t0 = time.perf_counter()
    for ep, seen in zip(eps, last):
        coverage(ep.positions, c.obs, *seen.shape)
    times['  of which coverage'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for seen in last:
        sample_queries(rng, seen, c.obs, c.queries)
    times['  of which query sampling'] = time.perf_counter() - t0

    print(f'per-episode stage cost over {episodes} episodes (us):')
    for name, total in times.items():
        print(f'  {name:36s} {1e6 * total / episodes:8.1f}')

    sizes = np.array([ep.tile_grid.shape for ep in eps])
    print(f'local map tiles per side: mean {sizes.mean():.2f}, '
          f'max {sizes.max()}; pixels per map mean '
          f'{(sizes.prod(axis=1) * c.tile ** 2).mean():.0f}')
    return eps, last


def query_statistics(config, bank, last_seen, episodes, seed=2):
    c = config
    rng = np.random.default_rng(seed)
    valid_counts, off_both, off_one, on = [], 0, 0, 0
    total = 0
    for ep, seen in zip(episodes, last_seen):
        queries, count = sample_queries(rng, seen, c.obs, c.queries)
        valid_counts.append(count)
        # phase of the query against the walk's lattice
        phase = (queries - ep.positions[0]) % c.step
        both = (phase != 0).all(axis=1).sum()
        one = ((phase != 0).sum(axis=1) == 1).sum()
        off_both += both
        off_one += one
        on += len(queries) - both - one
        total += len(queries)
    valid_counts = np.array(valid_counts)
    lattice_only = c.steps + 1
    print('query statistics:')
    print(f'  valid corners per episode: mean {valid_counts.mean():.0f}, '
          f'min {valid_counts.min()}, max {valid_counts.max()}')
    print(f'  episodes whose only valid corners are the {lattice_only} '
          f'observation windows themselves: '
          f'{(valid_counts <= lattice_only).mean():.4f}')
    print(f'  sampled queries off-lattice on both axes: {off_both / total:.3f}')
    print(f'  sampled queries off-lattice on one axis:  {off_one / total:.3f}')
    print(f'  sampled queries on the lattice (pure retrieval): '
          f'{on / total:.3f}')
    ages = []
    for ep, seen in zip(episodes, last_seen):
        q, _ = sample_queries(rng, seen, c.obs, c.queries)
        win = crop_windows(seen, q, c.obs)
        ages.append(c.steps - win)
    ages = np.concatenate(ages).ravel()
    print(f'  pixel age at query (steps since seen): mean {ages.mean():.1f}, '
          f'median {np.median(ages):.0f}, max {ages.max()}')


def numpy_throughput(config, bank, batch, seconds=5):
    rng = np.random.default_rng(3)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        make_batch(rng, config, bank, batch)
        n += 1
    elapsed = time.perf_counter() - t0
    print(f'numpy make_batch, batch {batch}: {n / elapsed:.1f} batches/s, '
          f'{n * batch / elapsed:.0f} episodes/s')


def loader_throughput(config, bank, batch, workers, batches=40):
    ds = StreamDataset(config, bank, batch)
    loader = DataLoader(ds, batch_size=None, num_workers=workers,
                        persistent_workers=workers > 0,
                        prefetch_factor=4 if workers > 0 else None)
    it = iter(loader)
    first = next(it)                                   # warm-up, spawn
    nbytes = sum(v.numel() * v.element_size() for v in first.values())
    t0 = time.perf_counter()
    for _ in range(batches):
        next(it)
    elapsed = time.perf_counter() - t0
    print(f'DataLoader workers={workers}: {batches / elapsed:.1f} batches/s, '
          f'{batches * batch / elapsed:.0f} episodes/s, '
          f'{nbytes / 1024:.0f} KiB/batch')
    return first


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', default='data/mnist.npz')
    p.add_argument('--obs', type=int, default=9)
    p.add_argument('--step', type=int, default=9)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--queries', type=int, default=8)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, nargs='*', default=[0, 2, 4, 8])
    args = p.parse_args()

    config = Config(obs=args.obs, step=args.step, steps=args.steps,
                    queries=args.queries)
    bank, _ = load_bank(args.bank, config.cutoff, dihedral=config.dihedral)
    print(config)
    print(f'bank {bank.shape}, {bank.nbytes / 2**20:.0f} MiB')

    eps, last = stage_breakdown(config, bank)
    query_statistics(config, bank, last, eps)
    numpy_throughput(config, bank, args.batch)
    for w in args.workers:
        first = loader_throughput(config, bank, args.batch, w)
    print('batch tensors:')
    for k, v in first.items():
        print(f'  {k:14s} {tuple(v.shape)} {v.dtype}')


if __name__ == '__main__':
    main()
