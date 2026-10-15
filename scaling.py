"""Parameter scaling for one model: enumerate its ladder, run it, fit it.

    python scaling.py --model X --dry            parameter count per rung
    python scaling.py --model X --local --steps 500 --sizes xs s
                                                 run rungs here, in sequence
    python scaling.py --model X --slurm --steps 20000
                                                 write a SLURM array for the
                                                 Torch cluster
    python scaling.py --model X --fit            power law from the runs

Add ``--lr-grid 1e-3 3e-4 1e-4`` to any run mode to train every rung at
each learning rate; ``--fit`` then takes the best learning rate per rung,
which is the standard treatment for a scaling comparison and removes the
confound of one rate suiting one width. Runs are named
``seed<seed>`` or ``seed<seed>_lr<lr>`` under ``results/<model>/<size>/``.

The first pass is a parameter sweep at fixed steps and batch, so every
rung sees the same episodes. A compute-matched sweep is a later variant
and is not what this script does.
"""
import argparse
import os
import subprocess
import sys

import mapping.tasks as tasks
from mapping.models import count_params
from mapping.results import best_per_size, fit_power_law, load_runs, table

CLUSTER = dict(account='torch_pr_1024_tandon_advanced',
               partition='h200_public', gpus=1, cpus=8, mem='32G',
               time='04:00:00', venv='/scratch/ax2348/venvs/mapping',
               workdir='/scratch/ax2348/mapping', max_concurrent=2)

SBATCH = """#!/bin/bash
#SBATCH --job-name=map-{task}-{model}{tag}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{last}%{max_concurrent}
#SBATCH --output={workdir}/jobs/logs/{task}-{model}-%A_%a.out
set -euo pipefail
SIZES=({sizes})
LRS=({lrs})
SIZE=${{SIZES[$(( SLURM_ARRAY_TASK_ID / ${{#LRS[@]}} ))]}}
LR=${{LRS[$(( SLURM_ARRAY_TASK_ID % ${{#LRS[@]}} ))]}}
LRFLAG=""; RUN=seed{seed}{tag}
if [ "$LR" != "default" ]; then LRFLAG="--lr $LR"; RUN=${{RUN}}_lr$LR; fi
source {venv}/bin/activate
cd {workdir}
python train.py --task {task} --model {model} --size $SIZE --run $RUN --steps {steps} \\
    --batch {batch} --seed {seed} --workers {workers} $LRFLAG {extra}
"""


def ladder(module, spec):
    for size, kwargs in module.SIZES.items():
        yield size, count_params(module.build(size, spec)), kwargs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='z2', choices=sorted(tasks.TASKS))
    p.add_argument('--model', required=True)
    p.add_argument('--sizes', nargs='*', default=None)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--extra', default='', help='passed through to train.py')
    p.add_argument('--tag', default='',
                   help='appended to run names and the job file, e.g. '
                        '_400k_b256, so budgets never collide')
    p.add_argument('--lr-grid', nargs='*', default=None,
                   help='learning rates to try at every rung')
    p.add_argument('--dry', action='store_true')
    p.add_argument('--local', action='store_true')
    p.add_argument('--slurm', action='store_true')
    p.add_argument('--fit', action='store_true')
    p.add_argument('--time', default=CLUSTER['time'])
    p.add_argument('--max-concurrent', type=int,
                   default=CLUSTER['max_concurrent'])
    args = p.parse_args()

    task = tasks.get(args.task)
    module = task['models'].get(args.model)
    spec = task['task'].TaskSpec()
    sizes = args.sizes or list(module.SIZES)
    lrs = args.lr_grid or ['default']

    if args.dry or not (args.local or args.slurm or args.fit):
        print(f'{args.model}: {getattr(module, "SCALING_RULE", "")}')
        lr_table = getattr(module, 'LR', {})
        print(f'{"size":6s} {"params":>12s}  {"lr":>7s}  kwargs')
        for size, n, kwargs in ladder(module, spec):
            if size in sizes:
                print(f'{size:6s} {n:12,d}  {lr_table.get(size, 3e-4):7.2g}'
                      f'  {kwargs}')

    if args.local:
        for size in sizes:
            for lr in lrs:
                run = f'seed{args.seed}'
                cmd = [sys.executable, 'train.py', '--task', args.task,
                       '--model', args.model,
                       '--size', size, '--steps', str(args.steps),
                       '--batch', str(args.batch), '--seed', str(args.seed),
                       '--workers', str(args.workers)]
                if lr != 'default':
                    cmd += ['--lr', lr]
                    run += f'_lr{lr}'
                cmd += ['--run', run + args.tag] + args.extra.split()
                print('>>', ' '.join(cmd), flush=True)
                subprocess.run(cmd, check=True)

    if args.slurm:
        os.makedirs('jobs/logs', exist_ok=True)
        path = (f'jobs/{args.model}{args.tag}.sbatch' if args.task == 'z2'
                else f'jobs/{args.task}_{args.model}{args.tag}.sbatch')
        with open(path, 'w') as f:
            f.write(SBATCH.format(task=args.task, model=args.model,
                                  tag=args.tag, sizes=' '.join(sizes),
                                  lrs=' '.join(lrs),
                                  last=len(sizes) * len(lrs) - 1,
                                  steps=args.steps, batch=args.batch,
                                  seed=args.seed, workers=args.workers,
                                  extra=args.extra,
                                  **{**CLUSTER, 'time': args.time,
                                     'max_concurrent': args.max_concurrent}))
        print(f'wrote {path} ({len(sizes) * len(lrs)} jobs, '
              f'{args.max_concurrent} at a time); after scripts/sync.sh '
              f'push:\n  ssh torch "cd {CLUSTER["workdir"]} && sbatch '
              f'{path}"')

    if args.fit:
        rows = best_per_size(load_runs(task['root'], args.model))
        if not rows:
            print('no runs found')
            return
        floor = rows[0]['floor']
        print(f'best run per rung (floor mse {floor["mse"]:.4f}, '
              f'fg_mae {floor["fg_mae"]:.4f})')
        print(table(rows))
        if len(rows) >= 3:
            for key in ('mse', 'fg_mae'):
                a, b, r2 = fit_power_law([r['params'] for r in rows],
                                         [r['final_eval'][key] for r in rows])
                print(f'{key}: loss = {a:.3g} * N^-{b:.3f}   (r2 {r2:.3f})')


if __name__ == '__main__':
    main()
