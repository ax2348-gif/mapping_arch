"""Training curves per model class: eval loss against training step, one
line per size, on one chart.

    python plot_curves.py                    every model -> results/plots/
    python plot_curves.py --model vanilla    one model
    python plot_curves.py --shuffled         add the permuted-direction eval
                                             as a dotted line per size

When a size has several runs (a learning-rate grid, or different step
budgets) the run with the lowest final eval mse is drawn, which is how the
sweep is read. Sizes are an ordered magnitude, so they take one hue from
light to dark. The constant floor is the dashed gray line.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from mapping.results import best_per_size, load_runs  # noqa: E402

# ordinal blue ramp, light -> dark, steps 250..700 of the reference palette
RAMP = ['#86b6ef', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
        '#184f95', '#104281', '#0d366b']
INK, MUTED, GRID = '#1f1f1f', '#6b6b6b', '#e4e4e4'


def curve(run):
    steps, mse, shuffled = [], [], []
    with open(os.path.join(run['path'], 'log.jsonl')) as f:
        for line in f:
            rec = json.loads(line)
            if 'eval' in rec:
                steps.append(rec['step'])
                mse.append(rec['eval']['mse'])
                shuffled.append(rec.get('eval_shuffled', {}).get('mse'))
    return steps, mse, shuffled


def ramp(n):
    if n <= 1:
        return [RAMP[4]]
    idx = [round(i * (len(RAMP) - 1) / (n - 1)) for i in range(n)]
    return [RAMP[i] for i in idx]


def draw(ax, runs, model, shuffled=False):
    runs = sorted(runs, key=lambda r: r['params'])
    colors = ramp(len(runs))
    floor = runs[0]['floor']['mse']
    ax.axhline(floor, color=MUTED, linestyle='--', linewidth=1.2, zorder=1)
    ax.text(0.01, floor, 'constant floor', color=MUTED, fontsize=8,
            va='bottom', ha='left',
            transform=ax.get_yaxis_transform())
    for r, color in zip(runs, colors):
        steps, mse, shf = curve(r)
        label = f'{r["size"]} ({r["params"] / 1e6:.2g}M)'
        ax.plot(steps, mse, color=color, linewidth=2, label=label, zorder=3)
        ax.text(steps[-1], mse[-1], f' {r["size"]}', color=INK, fontsize=8,
                va='center')
        if shuffled and all(s is not None for s in shf):
            ax.plot(steps, shf, color=color, linewidth=1, linestyle=':',
                    zorder=2)
    ax.set_title(model, loc='left', fontsize=11, color=INK, fontweight='bold')
    ax.set_xlabel('training step', color=MUTED)
    ax.set_ylabel('eval mse (held-out, test bank)', color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, title='size (params)',
              title_fontsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--root', default='results')
    p.add_argument('--out', default=None, help='default results/plots/')
    p.add_argument('--shuffled', action='store_true')
    a = p.parse_args()
    out_dir = a.out or os.path.join(a.root, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    rows = best_per_size(load_runs(a.root, a.model))
    by_model = {}
    for r in rows:
        by_model.setdefault(r['model'], []).append(r)
    if not by_model:
        print('no runs found')
        return

    for model, runs in by_model.items():
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
        draw(ax, runs, model, a.shuffled)
        fig.tight_layout()
        path = os.path.join(out_dir, f'{model}_curves.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {path} ({len(runs)} sizes)')

    if len(by_model) > 1:
        n = len(by_model)
        cols = min(2, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 4.5 * rows_n),
                                 dpi=130, squeeze=False)
        lo = min(min(curve(r)[1]) for runs in by_model.values() for r in runs)
        hi = max(r['floor']['mse'] for runs in by_model.values()
                 for r in runs) * 1.05
        if a.shuffled:
            hi = max(hi, max(max(s for s in curve(r)[2] if s is not None)
                             for runs in by_model.values() for r in runs
                             if any(curve(r)[2])) * 1.02)
        for ax, (model, runs) in zip(axes.flat, by_model.items()):
            draw(ax, runs, model, a.shuffled)
            ax.set_ylim(lo * 0.95, hi)
        for ax in list(axes.flat)[n:]:
            ax.axis('off')
        fig.tight_layout()
        path = os.path.join(out_dir, 'curves.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {path} (all models, shared y axis)')


if __name__ == '__main__':
    main()
