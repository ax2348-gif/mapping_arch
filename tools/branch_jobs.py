"""Write a SLURM array of warmup-stable-decay cooldown branches.

    python tools/branch_jobs.py --task se2d --model hybrid4 --sizes xs s m \
        --parent seed0_400k_qt_wsd_small --budgets 62500 125000 250000 \
        --extra "task.query_times=uniform" --out jobs/se2d_hybrid4_small_cd.sbatch

A long warmup-stable-decay run that saved ``step<k>.pt`` every 50k steps
gives a cheap loss point at any shorter budget S: branch from the
checkpoint at k = 0.8 S and decay to zero over the last 20% (train.py
``schedule=wsd run.branch=...``). The branch is the same recipe as a fresh
S-step run (Hägele et al. 2024), at a cost of 0.2 S steps. Each array task
is one (size, budget); run names are ``<parent>_cd<S/1000>k``. Submit with
``sbatch --dependency=afterok:<parent job id> <out>``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mapping.tasks as tasks  # noqa: E402

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=map-{task}-{model}-cd
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time={time}
#SBATCH --array=0-{last}%{concurrent}
#SBATCH --output=/scratch/ax2348/mapping/jobs/logs/{task}-{model}-cd-%A_%a.out
set -euo pipefail
SIZES=({sizes})
BUDGETS=({budgets})
SIZE=${{SIZES[$(( SLURM_ARRAY_TASK_ID / ${{#BUDGETS[@]}} ))]}}
S=${{BUDGETS[$(( SLURM_ARRAY_TASK_ID % ${{#BUDGETS[@]}} ))]}}
K=$(( S * 4 / 5 ))
PARENT=results/{root}/{model}/$SIZE/{parent}
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping
python -c "import hydra, omegaconf; print('hydra-core', hydra.__version__, 'omegaconf', omegaconf.__version__)" \\
    || {{ echo "this venv lacks the config tree's dependencies: pip install -r requirements.txt" >&2; exit 1; }}
test -f $PARENT/step$K.pt
python run.py model={task}/{model} model.size=$SIZE run.steps=$S \\
    schedule=wsd run.branch=$PARENT/step$K.pt run.name={parent}_cd$(( S / 1000 ))k \\
    optim.batch=64 run.seed={seed} {workers} eval.every=2500 eval.log_every=1000 {extra}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='se2d', choices=sorted(tasks.ALL))
    p.add_argument('--model', required=True)
    p.add_argument('--sizes', nargs='+', required=True)
    p.add_argument('--parent', required=True, help='parent run name')
    p.add_argument('--budgets', nargs='+', type=int, default=[62500, 125000, 250000])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--extra', default='', help='Hydra overrides passed through to run.py')
    p.add_argument('--time', default='12:00:00')
    p.add_argument('--concurrent', type=int, default=3)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    for s in a.budgets:
        assert (s * 4) % 5 == 0 and (s * 4 // 5) % 50000 == 0, \
            f'budget {s}: branch point {s * 4 / 5} must be a saved checkpoint (multiple of 50k)'
    if any(x.startswith('--') for x in a.extra.split()):
        raise SystemExit(f'--extra {a.extra!r}: jobs launch run.py, so --extra takes '
                         f'Hydra overrides (task.query_times=uniform), not train.py flags')
    task = tasks.get(a.task)
    root = task['root'][len('results'):].strip('/')
    workers = '' if task.get('gpu_data') else 'task.generator.workers=14'
    text = TEMPLATE.format(task=a.task, model=a.model, sizes=' '.join(a.sizes),
                           budgets=' '.join(map(str, a.budgets)),
                           last=len(a.sizes) * len(a.budgets) - 1,
                           concurrent=a.concurrent, time=a.time, parent=a.parent,
                           root=root, seed=a.seed, extra=a.extra, workers=workers)
    text = text.replace('results//', 'results/')
    open(a.out, 'w').write(text)
    print(f'wrote {a.out}: {len(a.sizes) * len(a.budgets)} branches')


if __name__ == '__main__':
    main()
