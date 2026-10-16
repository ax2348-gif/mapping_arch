"""Every run under results/ as one table, and power-law fits per model.

    python summarize.py                 all z2 runs
    python summarize.py --task se2d      all se2d runs
    python summarize.py --model lstm    one model
    python summarize.py --fit           best run per rung, fitted per model
"""
import argparse

import mapping.tasks as tasks
from mapping.results import best_per_size, fit_power_law, load_runs, table


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='z2', choices=sorted(tasks.TASKS))
    p.add_argument('--model', default=None)
    p.add_argument('--root', default=None, help="default: the task's")
    p.add_argument('--fit', action='store_true')
    a = p.parse_args()
    a.root = a.root or tasks.get(a.task)['root']
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
