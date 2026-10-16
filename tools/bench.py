"""Throughput for sizing long runs: training steps per second per model,
rung and batch, and the data loader's episodes per second.

    python tools/bench.py --task se2 --models vanilla4 hybrid4 \
        --sizes m l xl xxl --batches 64 128 256 --query-times uniform

Each configuration builds the model, runs a few warm-up steps, then times
``--steps`` full training steps (forward, backward, AdamW) in bf16 on
pre-generated real batches, so the loader does not limit it. The loader is
timed separately with several worker counts. Prints one line per
configuration and the peak GPU memory.
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.loader import eval_set, load_banks, train_loader  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='se2')
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--sizes', nargs='+', required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[64, 128, 256])
    p.add_argument('--query-times', default='uniform')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--loader-workers', nargs='*', type=int, default=[4, 8, 16])
    a = p.parse_args()
    dev = 'cuda'
    t = tasks.get(a.task)
    config, spec = tasks.config_and_spec(a.task, 32, 8, 0, a.query_times)
    train_bank, _ = load_banks('data/mnist.npz', config)
    prepare = t['task'].prepare
    print(torch.cuda.get_device_name(0), flush=True)

    for bs in a.batches:
        batches = eval_set(config, train_bank, bs, 4, 0)
        for name in a.models:
            for size in a.sizes:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    model = t['models'].get(name).build(size, spec).to(dev)
                except Exception as e:
                    print(f'{name} {size} b{bs}: build failed {e}')
                    continue
                n = sum(x.numel() for x in model.parameters())
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

                def step(i):
                    obs, odo, q, tgt, qt = prepare(batches[i % 4], dev)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = model(obs, odo, q, query_time=qt)
                    loss = torch.nn.functional.mse_loss(pred.float(), tgt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    for i in range(3):
                        step(i)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for i in range(a.steps):
                        step(i)
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) / a.steps
                    mem = torch.cuda.max_memory_allocated() / 2**30
                    print(f'{name:9s} {size:4s} {n:12,d} params  batch {bs:4d}: '
                          f'{1 / dt:7.1f} steps/s  {bs / dt:9.0f} episodes/s  '
                          f'peak {mem:5.1f} GiB', flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f'{name} {size} b{bs}: out of memory', flush=True)
                del model, opt

    for w in a.loader_workers:
        loader = train_loader(config, train_bank, 256, w)
        it = iter(loader)
        for _ in range(3):
            next(it)
        t0 = time.perf_counter()
        for _ in range(20):
            next(it)
        dt = time.perf_counter() - t0
        print(f'loader {w:2d} workers, batch 256: {20 * 256 / dt:8.0f} episodes/s',
              flush=True)
        del it, loader


if __name__ == '__main__':
    main()
