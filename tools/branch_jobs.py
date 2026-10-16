"""Write a SLURM array of warmup-stable-decay cooldown branches.

    python tools/branch_jobs.py --task se2 --model hybrid4 --sizes xs s m \
        --parent seed0_400k_qt_wsd_small --budgets 62500 125000 250000 \
        --extra "--query-times uniform" --out jobs/se2_hybrid4_small_cd.sbatch

A long warmup-stable-decay run that saved ``step<k>.pt`` every 50k steps
gives a cheap loss point at any shorter budget S: branch from the
checkpoint at k = 0.8 S and decay to zero over the last 20% (train.py
``--schedule wsd --branch``). The branch is the same recipe as a fresh
S-step run (Hägele et al. 2024), at a cost of 0.2 S steps. Each array task
is one (size, budget); run names are ``<parent>_cd<S/1000>k``. Submit with
``sbatch --dependency=afterok:<parent job id> <out>``.
"""
import argparse

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
test -f $PARENT/step$K.pt
python train.py --task {task} --model {model} --size $SIZE --steps $S \\
    --schedule wsd --branch $PARENT/step$K.pt --run {parent}_cd$(( S / 1000 ))k \\
    --batch 64 --seed {seed} --workers 14 --eval-every 2500 --log-every 1000 {extra}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='se2')
    p.add_argument('--model', required=True)
    p.add_argument('--sizes', nargs='+', required=True)
    p.add_argument('--parent', required=True, help='parent run name')
    p.add_argument('--budgets', nargs='+', type=int, default=[62500, 125000, 250000])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--extra', default='')
    p.add_argument('--time', default='12:00:00')
    p.add_argument('--concurrent', type=int, default=3)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    for s in a.budgets:
        assert (s * 4) % 5 == 0 and (s * 4 // 5) % 50000 == 0, \
            f'budget {s}: branch point {s * 4 / 5} must be a saved checkpoint (multiple of 50k)'
    root = 'se2' if a.task == 'se2' else '.'
    root = root if root != '.' else ''
    text = TEMPLATE.format(task=a.task, model=a.model, sizes=' '.join(a.sizes),
                           budgets=' '.join(map(str, a.budgets)),
                           last=len(a.sizes) * len(a.budgets) - 1,
                           concurrent=a.concurrent, time=a.time, parent=a.parent,
                           root=root, seed=a.seed, extra=a.extra)
    text = text.replace('results//', 'results/')
    open(a.out, 'w').write(text)
    print(f'wrote {a.out}: {len(a.sizes) * len(a.budgets)} branches')


if __name__ == '__main__':
    main()
