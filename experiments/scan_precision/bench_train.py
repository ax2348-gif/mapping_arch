"""Training-step throughput, ``model.scan_precision`` tf32 vs ieee: model
forward + backward + AdamW under bf16 autocast, batch 64, 8 queries at
uniform times, random inputs (no data generator), for each model in CASES.
The two settings alternate over ROUNDS rounds of STEPS steps (the local GPU
is shared with a training job; the median round is reported).

    python experiments/scan_precision/bench_train.py
"""
import json
import os
import statistics
import sys
import time
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mapping.tasks as tasks  # noqa: E402

CASES = [('se2c', 'hybrid4c_causal', 's', 'inert'), ('se2c', 'hybrid4c', 's', 'mixed')]
WARMUP, STEPS, ROUNDS = 15, 30, 3
OUT = os.path.join(ROOT, 'results', 'scan_precision', 'bench_train.json')


def main():
    out = {}
    for task, name, size, mode in CASES:
        module = tasks.get(task)['models'].get(name)
        spec = tasks.get('se2c')['task'].TaskSpec()
        cls = type(module.build(next(iter(module.SIZES)), spec))
        b, q = 64, 8
        g = torch.Generator().manual_seed(0)
        obs = torch.rand(b, 33, 15, 15, generator=g).cuda()
        odo = torch.cat([torch.randn(b, 32, 2, generator=g) * 6,
                         (torch.rand(b, 32, 1, generator=g) * 2 - 1) * 3.14], -1).cuda()
        pose = torch.cat([torch.randn(b, q, 2, generator=g) * 20,
                          (torch.rand(b, q, 1, generator=g) * 2 - 1) * 3.14], -1).cuda()
        qt = torch.randint(0, 33, (b, q), generator=g).cuda()
        tgt = torch.rand(b, q, 15, 15, generator=g).cuda()
        runs = {}
        for prec in ('tf32', 'ieee'):
            torch.manual_seed(0)
            net = cls(spec, **module.SIZES[size], query_mode=mode, scan_precision=prec).cuda()
            opt = torch.optim.AdamW(net.parameters(), 3e-4)

            def step(net=net, opt=opt):
                with torch.autocast('cuda', torch.bfloat16):
                    loss = F.mse_loss(net(obs, odo, pose, query_time=qt).float(), tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            for _ in range(WARMUP):
                step()
            runs[prec] = step
        rates = {p: [] for p in runs}
        for _ in range(ROUNDS):
            for p, step in runs.items():
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(STEPS):
                    step()
                torch.cuda.synchronize()
                rates[p].append(STEPS / (time.time() - t))
        key = f'{task}/{name}/{size}/{mode}'
        out[key] = {p: dict(it_s=statistics.median(r), rounds=r) for p, r in rates.items()}
        out[key]['ieee_over_tf32'] = out[key]['ieee']['it_s'] / out[key]['tf32']['it_s']
        print(key, {p: f"{v['it_s']:.2f} it/s" for p, v in out[key].items() if p != 'ieee_over_tf32'},
              f"ratio {out[key]['ieee_over_tf32']:.3f}", flush=True)
        del runs
        torch.cuda.empty_cache()
    out['device'] = torch.cuda.get_device_name()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
