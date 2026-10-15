"""Reading runs back: every ``final.json`` under ``results/`` as rows, and
a power-law fit over a model's ladder.
"""
import glob
import json
import os

import numpy as np

ROOT = 'results'


def load_runs(root=ROOT, model=None):
    """One dict per run directory that holds a ``final.json``, with the
    run name added, sorted by model, parameters, run."""
    pattern = os.path.join(root, model or '*', '*', '*', 'final.json')
    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            r = json.load(f)
        r['run'] = os.path.basename(os.path.dirname(path))
        r['path'] = os.path.dirname(path)
        rows.append(r)
    rows.sort(key=lambda r: (r['model'], r['params'], r['run']))
    return rows


def best_per_size(rows):
    """Keep, for each (model, size), the run with the lowest final eval
    mse: the way a learning-rate grid is read."""
    best = {}
    for r in rows:
        key = (r['model'], r['size'], r.get('query_times', 'last'))
        if key not in best or (r['final_eval']['mse']
                               < best[key]['final_eval']['mse']):
            best[key] = r
    return sorted(best.values(), key=lambda r: (r['model'], r['params']))


def fit_power_law(params, losses):
    """``loss = a * params ** -b`` by least squares in log-log; returns
    ``(a, b, r2)``."""
    x = np.log(np.asarray(params, float))
    y = np.log(np.asarray(losses, float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(np.exp(intercept)), float(-slope), float(r2)


def table(rows, columns=('model', 'size', 'run', 'params', 'steps', 'lr',
                         'qt', 'mse', 'last', 'shuffled', 'fg_mae', 'fg<0.1',
                         'it/s')):
    """Rows as aligned text."""
    def cell(r, c):
        e, s = r['final_eval'], r.get('final_eval_shuffled', {})
        return {
            'model': r['model'], 'size': r['size'], 'run': r['run'],
            'params': f'{r["params"]:,}', 'steps': str(r['steps']),
            'lr': f'{r["args"]["lr"]:.2g}', 'mse': f'{e["mse"]:.4f}',
            'qt': r.get('query_times', 'last')[:4],
            'last': f'{e["mse_last"]:.4f}' if 'mse_last' in e else '-',
            'shuffled': f'{s["mse"]:.4f}' if s else '-',
            'fg_mae': f'{e["fg_mae"]:.3f}',
            'fg<0.1': f'{e["fg_within_0.1"]:.3f}',
            'it/s': f'{r["steps"] / r["elapsed_s"]:.0f}',
        }[c]
    grid = [[cell(r, c) for c in columns] for r in rows]
    widths = [max(len(c), *(len(g[i]) for g in grid)) if grid else len(c)
              for i, c in enumerate(columns)]
    lines = ['  '.join(c.ljust(w) for c, w in zip(columns, widths))]
    for g in grid:
        lines.append('  '.join(v.ljust(w) for v, w in zip(g, widths)))
    return '\n'.join(lines)
