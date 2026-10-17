"""Held-out eval of se2d checkpoints, split by composition span, on any walk
path.

    python experiments/se2d_saw/eval_split.py OUT.json PATH [steps=N] [offset_max=M] CKPT [CKPT ...]

``steps=N`` evaluates on N-step episodes and ``offset_max=M`` pins the
spec's offset scale; the model is then rebuilt at that spec and given the
checkpoint's weights (as ``run.init_from`` does), for zero-shot length
transfer.

PATH (mapping.data.PATHS) replaces the checkpoint's own walk path, so a
model trained on one path is evaluated on another; everything else of the
generator config comes from the checkpoint's args. The batches are the
trainer's held-out set for that config (seed [seed, EVAL_SEED_TAG], 16 x 64
episodes), so on the checkpoint's own path the overall mse is its final
eval. Each query is binned by ``span_needed`` (spans.py: near < 4, mid
4..11, far >= 12 steps) and by whether its target was re-observed; the
same with the odometry permuted in time (the trainer's shuffled control,
same generator seed).
"""
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mapping.tasks as tasks  # noqa: E402
import train  # noqa: E402
from mapping.checkpoint import build_from_arch, load_checkpoint  # noqa: E402
from mapping.data import make_batch  # noqa: E402
from mapping.loader import EVAL_SEED_TAG, load_banks  # noqa: E402
from spans import BINS, NAMES, query_stats  # noqa: E402

OUT, PATH = sys.argv[1], sys.argv[2]
OPTS = dict(a.split('=', 1) for a in sys.argv[3:] if '=' in a)
CKPTS = [a for a in sys.argv[3:] if '=' not in a]
BATCH, BATCHES = 64, 16


def run(ck):
    net, args, spec, task = load_checkpoint(ck, 'cuda')
    if OPTS:
        args = dict(args, episode_steps=int(OPTS.get('steps', spec.steps)))
        if 'offset_max' in OPTS:
            args['offset_max'] = int(OPTS['offset_max'])
        _, spec = tasks.config_and_spec(task, args['episode_steps'], spec.queries,
                                        args['seed'], args.get('query_times', 'last'),
                                        args.get('step_px', 9), args.get('odometry_dropout', 0.0),
                                        args.get('texture', 'mnist'), PATH, args.get('offset_max'))
        state = net.state_dict()
        net = build_from_arch(torch.load(ck, map_location='cpu', weights_only=False)['arch'],
                              spec).to('cuda').eval()
        net.load_state_dict(state)
    config = replace(tasks.config_from_args(task, spec, args), path=PATH)
    _, bank = load_banks(args.get('bank') or 'data/mnist.npz', config,
                         args.get('texture', 'mnist'))
    prepare = tasks.get(task)['task'].prepare
    rng = np.random.default_rng([args['seed'], EVAL_SEED_TAG])
    err, err_sh, q = [], [], []
    gen = torch.Generator().manual_seed(0)
    autocast = torch.autocast('cuda', torch.bfloat16)
    with torch.no_grad():
        for _ in range(BATCHES):
            b = make_batch(rng, config, bank, BATCH)
            q.append(query_stats(b, config.step, config.obs))
            b.pop('episodes')
            b = {k: torch.from_numpy(v) for k, v in b.items()}
            obs, odo, pose, target, qt = prepare(b, 'cuda')
            perm = torch.argsort(torch.rand(odo.shape[:2], generator=gen),
                                 dim=1).to('cuda')
            for o, out in ((odo, err), (train.permute_steps(odo, perm), err_sh)):
                with autocast:
                    pred = net(obs, o, pose, query_time=qt)
                out.append((pred.float() - target).pow(2).mean((-1, -2))
                           .flatten().cpu().numpy())
    err, err_sh = np.concatenate(err), np.concatenate(err_sh)
    q = {k: np.concatenate([x[k] for x in q]) for k in q[0]}
    res = {'path': PATH, 'trained_path': args.get('path', 'walk'), 'opts': OPTS,
           'steps': config.steps, 'offset_max': spec.offset_max,
           'mse': float(err.mean()), 'mse_shuffled': float(err_sh.mean()),
           'queries': int(err.size)}
    bins = np.digitize(q['span_needed'], BINS[1:-1])
    for i, name in enumerate(NAMES):
        m = bins == i
        res[f'span_{name}'] = {'frac': float(m.mean()),
                               'mse': float(err[m].mean()) if m.any() else None,
                               'mse_shuffled': float(err_sh[m].mean()) if m.any() else None}
    for name, m in (('reobserved', q['reobserved']), ('not_reobserved', ~q['reobserved'])):
        res[name] = {'frac': float(m.mean()),
                     'mse': float(err[m].mean()) if m.any() else None}
    return res


out = {ck: run(ck) for ck in CKPTS}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
