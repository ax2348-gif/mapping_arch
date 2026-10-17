"""Held-out eval mse of checkpoints under the code at ROOT (bf16 autocast,
as the trainer evaluates), to compare attention implementations.

    python experiments/inert_queries/eval_ckpts.py ROOT OUT.json CKPT [CKPT ...]
"""
import json
import sys

root = sys.argv[1]
sys.path.insert(0, root)
import torch  # noqa: E402

import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402

out = {}
for ck in sys.argv[3:]:
    net, args, spec, task = load_checkpoint(ck, 'cuda')
    config = tasks.config_from_args(task, spec, args)
    _, bank = load_banks(args.get('bank') or 'data/mnist.npz', config, args.get('texture', 'mnist'))
    prepare = tasks.get(task)['task'].prepare
    se, n = 0.0, 0
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        for b in eval_set(config, bank, 64, 16, args['seed']):
            obs, odo, pose, target, qt = prepare(b, 'cuda')
            se += (net(obs, odo, pose, query_time=qt).float() - target).pow(2).sum().item()
            n += target.numel()
    out[ck] = se / n
json.dump(out, open(sys.argv[2], 'w'))
