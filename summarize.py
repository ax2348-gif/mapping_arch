"""Every run of a task as one table, and power-law fits per model.

    python summarize.py                 all z2 runs, under results/z2/
    python summarize.py --task se2d      all se2d runs (se2c likewise)
    python summarize.py --task se2       the old se2 runs, under results/se2/
    python summarize.py --model lstm    one model
    python summarize.py --fit           best run per rung, fitted per model
"""
import argparse

import mapping.tasks as tasks
from mapping.results import best_per_size, fit_power_law, load_runs, table


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='z2',
                   choices=sorted(tasks.ALL) + sorted(tasks.ALIASES),
                   help="which task's results root; an alias (se2) reads its old runs")
    p.add_argument('--model', default=None)
    p.add_argument('--root', default=None,
                   help="the task's run directories; default results/<task> (tasks.results_dir)")
    p.add_argument('--fit', action='store_true')
    a = p.parse_args()
    a.root = a.root or tasks.results_dir(a.task)
    rows = load_runs(a.root, a.model)
    if not rows:
        print('no runs found')
        return
    if not a.fit:
        print(table(rows))
        return
    rows = best_per_size(rows)
    print(table(rows))
    by_model = {}
    for r in rows:
        by_model.setdefault(r['model'], []).append(r)
    for model, group in by_model.items():
        if len(group) < 3:
            continue
        for key in ('mse', 'fg_mae'):
            a_, b, r2 = fit_power_law([r['params'] for r in group],
                                      [r['final_eval'][key] for r in group])
            print(f'{model} {key}: {a_:.3g} * N^-{b:.3f}  (r2 {r2:.3f}, '
                  f'{len(group)} rungs)')


if __name__ == '__main__':
    main()
