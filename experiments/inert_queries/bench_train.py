"""Training-step throughput (model forward + backward + AdamW, bf16, batch
64, 8 queries at uniform times; no data loader) under the code at ROOT.

    python experiments/inert_queries/bench_train.py ROOT OUT.json [sizes...]
"""
import json
import sys
import time
import warnings

root = sys.argv[1]
sys.path.insert(0, root)
warnings.filterwarnings('ignore')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mapping.tasks as tasks  # noqa: E402

spec = tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')[1]
reg = tasks.get('se2d')['models']
CASES = [('mixed', 'hybrid4', 'mixed'), ('inert', 'hybrid4', 'inert'),
         ('causal', 'hybrid4_causal', 'mixed'), ('causal-inert', 'hybrid4_causal', 'inert')]
out = {}
for size in sys.argv[3:]:
    for label, name, mode in CASES:
        module = reg.get(name)
        torch.manual_seed(0)
        net = type(module.build('xs', spec))(spec, **module.SIZES[size], query_mode=mode).cuda()
        opt = torch.optim.AdamW(net.parameters(), 3e-4)
        b, q = 64, 8
        g = torch.Generator().manual_seed(0)
        obs = torch.rand(b, 33, 9, 9, generator=g).cuda()
        odo = torch.randint(0, 16, (b, 32), generator=g).cuda()
        pose = torch.cat([torch.randint(-80, 80, (b, q, 2), generator=g),
                          torch.randint(0, 4, (b, q, 1), generator=g)], -1).cuda()
        qt = torch.randint(0, 33, (b, q), generator=g).cuda()
        tgt = torch.rand(b, q, 9, 9, generator=g).cuda()

        def step():
            with torch.autocast('cuda', torch.bfloat16):
                loss = F.mse_loss(net(obs, odo, pose, query_time=qt).float(), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for _ in range(25):
            step()
        n = 60 if size in ('xs', 's') else 30
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        out[f'{size}/{label}'] = n / (time.time() - t)
        print(size, label, f'{out[f"{size}/{label}"]:.1f} it/s', flush=True)
        del net, opt
        torch.cuda.empty_cache()
json.dump(out, open(sys.argv[2], 'w'))
