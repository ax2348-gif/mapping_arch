"""Eval cost of many queries on one history (batch 1, bf16, no grad):
peak memory and time per forward, under the code at ROOT.

    python experiments/inert_queries/bench_eval.py ROOT OUT.json SIZE MODEL MODE Q [Q ...]
"""
import json
import sys
import time
import warnings

root = sys.argv[1]
sys.path.insert(0, root)
warnings.filterwarnings('ignore')
import torch  # noqa: E402

import mapping.tasks as tasks  # noqa: E402

spec = tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')[1]
size, name, mode = sys.argv[3:6]
module = tasks.get('se2d')['models'].get(name)
torch.manual_seed(0)
net = type(module.build('xs', spec))(spec, **module.SIZES[size], query_mode=mode).cuda().eval()
out = {}
for q in map(int, sys.argv[6:]):
    g = torch.Generator().manual_seed(0)
    obs = torch.rand(1, 33, 9, 9, generator=g).cuda()
    odo = torch.randint(0, 16, (1, 32), generator=g).cuda()
    pose = torch.cat([torch.randint(-80, 80, (1, q, 2), generator=g),
                      torch.randint(0, 4, (1, q, 1), generator=g)], -1).cuda()
    qt = torch.randint(0, 33, (1, q), generator=g).cuda()

    def fwd():
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            return net(obs, odo, pose, query_time=qt)
    try:
        fwd()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t = time.time()
        for _ in range(5):
            fwd()
        torch.cuda.synchronize()
        out[q] = {'ms': (time.time() - t) / 5 * 1e3,
                  'peak_mib': (torch.cuda.max_memory_allocated() - base) / 2**20}
    except torch.OutOfMemoryError as e:
        out[q] = {'error': 'OOM ' + str(e)[:80]}
    torch.cuda.empty_cache()
    print(size, name, mode, q, out[q], flush=True)
json.dump(out, open(sys.argv[2], 'w'))
