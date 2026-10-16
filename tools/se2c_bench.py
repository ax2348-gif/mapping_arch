"""Throughput of the se2c path: the GPU generator alone, and full training
steps (forward, backward, AdamW, bf16) of the se2c models, on a fixed
batch and with the batch generated inside the step, as train.py does.

    python tools/se2c_bench.py --models hybrid4c vanilla4c --sizes xs s

``--patch 5`` measures the 5 x 5-patch tokeniser (9 tokens per view)
instead of the default 3 x 3 (25), for the cost comparison only.
"""
import argparse
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.se2c.generator import Config, make_batch  # noqa: E402


def timed(fn, n):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(n):
        fn(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['hybrid4c', 'vanilla4c'])
    p.add_argument('--sizes', nargs='+', default=['xs', 's'])
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--patch', type=int, default=3)
    a = p.parse_args()
    dev = 'cuda'
    t = tasks.get('se2c')
    spec = t['task'].TaskSpec()
    c = Config()
    print(torch.cuda.get_device_name(0), flush=True)
    g = timed(lambda i=0: make_batch(c, i, dev, a.batch), a.steps)
    print(f'generator: {g * 1e3:.1f} ms per batch of {a.batch}', flush=True)
    fixed = make_batch(c, 0, dev, a.batch)
    for name in a.models:
        module = t['models'].get(name)
        for size in a.sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            kw = dict(module.SIZES[size])
            if a.patch != 3:
                kw['patch'] = a.patch
            cls = {'hybrid4c': 'Hybrid4c', 'vanilla4c': 'Vanilla4c'}[name]
            model = getattr(module, cls)(spec, **kw).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

            def step(i=0, gen=False):
                b = make_batch(c, i, dev, a.batch) if gen else fixed
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(b['observations'], b['odometry'],
                                 b['query_pose'], query_time=b['query_time'])
                loss = F.mse_loss(pred.float(), b['target'])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            s_fixed = timed(lambda i=0: step(i), a.steps)
            s_gen = timed(lambda i=0: step(i, True), a.steps)
            n = sum(x.numel() for x in model.parameters())
            print(f'{name} {size} patch {a.patch}: {n:,} params, '
                  f'{1 / s_fixed:.1f} steps/s fixed batch, {1 / s_gen:.1f} '
                  f'steps/s with generation, peak '
                  f'{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB',
                  flush=True)


if __name__ == '__main__':
    main()
