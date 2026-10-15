"""Draw the prediction grid for any checkpoint.

    python render.py results/vanilla/xs/seed0/final.pt [--out grid.png]

``train.py`` already writes ``grid.png`` next to every checkpoint; this
redraws one, for instance with more rows or a different sample.
"""
import argparse
import os

import torch

import mapping.tasks as tasks
from mapping.checkpoint import load_checkpoint
from mapping.loader import eval_set, load_banks
from mapping.render import render_grid, run_title


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--out', default=None, help='default: grid.png beside it')
    p.add_argument('--rows', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    model, args, spec, task = load_checkpoint(a.checkpoint, a.device)
    config, _ = tasks.config_and_spec(task, spec.steps, spec.queries,
                                      args['seed'])
    _, eval_bank = load_banks(args['bank'], config)
    batches = eval_set(config, eval_bank, 64, 2, args['seed'])
    out = a.out or os.path.join(os.path.dirname(a.checkpoint), 'grid.png')
    title = ''
    summary = os.path.join(os.path.dirname(a.checkpoint), 'final.json')
    if os.path.exists(summary):
        import json
        with open(summary) as f:
            final = json.load(f)
        final.setdefault('run', os.path.basename(os.path.dirname(summary)))
        title = run_title(final)
    render_grid(model, batches, spec.view, out, rows=a.rows, seed=a.seed,
                title=title, prepare=tasks.get(task)['task'].prepare)
    print(f'{args["model"]}/{args["size"]}: wrote {out}')


if __name__ == '__main__':
    main()
