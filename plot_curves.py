"""Learning curves per model class: eval mse against training step, one
line per size, on one chart.

    python plot_curves.py                    every model
    python plot_curves.py --model vanilla    one model
    python plot_curves.py --shuffled         add the permuted-direction eval
                                             as a dotted line per size

Writes results/<model>/curves.png for each model and results/curves.png
with every model on a shared axis. When a size has several runs (a
learning-rate grid, or different step budgets) the run with the lowest
final eval mse is drawn, which is how the sweep is read. Every run also
has its own curve.png, written by train.py.
"""
import argparse
import os

import matplotlib.pyplot as plt

from mapping.plot import draw_model, plot_model
from mapping.results import best_per_size, load_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--root', default='results')
    p.add_argument('--shuffled', action='store_true')
    a = p.parse_args()

    rows = best_per_size(load_runs(a.root, a.model))
    by_model = {}
    for r in rows:
        by_model.setdefault(r['model'], []).append(r)
    if not by_model:
        print('no runs found')
        return
    for model, runs in by_model.items():
        out = plot_model(model, runs,
                         os.path.join(a.root, model, 'curves.png'),
                         a.shuffled)
        print(f'wrote {out} ({len(runs)} sizes)')
    if len(by_model) > 1:
        n, cols = len(by_model), 2
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, dpi=130,
                                 figsize=(7 * cols, 4.5 * rows_n),
                                 squeeze=False)
        for ax, (model, runs) in zip(axes.flat, by_model.items()):
            draw_model(ax, runs, model, a.shuffled)
        lo = min(ax.get_ylim()[0] for ax in axes.flat[:n])
        hi = max(ax.get_ylim()[1] for ax in axes.flat[:n])
        for ax in axes.flat[:n]:
            ax.set_ylim(lo, hi)
        for ax in list(axes.flat)[n:]:
            ax.axis('off')
        fig.tight_layout()
        out = os.path.join(a.root, 'curves.png')
        fig.savefig(out)
        plt.close(fig)
        print(f'wrote {out} (all models, shared y axis)')


if __name__ == '__main__':
    main()
