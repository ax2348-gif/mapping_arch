"""Learning curves from run logs.

``plot_run`` draws one run (eval mse and the permuted-direction eval against
step, with the constant floor and the train loss) to ``curve.png`` in its
directory; ``train.py`` calls it at the end of every run. ``plot_model``
draws every size of one model on one chart to ``curves.png`` in the
model's directory, best run per size.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# ordinal blue ramp, light -> dark, steps 250..700 of the reference palette
RAMP = ['#86b6ef', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
        '#184f95', '#104281', '#0d366b']
INK, MUTED, GRID = '#1f1f1f', '#6b6b6b', '#e4e4e4'


def read_log(run_dir):
    """``(steps, train_mse, eval_steps, eval_mse, shuffled_mse)``."""
    steps, train, es, ev, sh = [], [], [], [], []
    with open(os.path.join(run_dir, 'log.jsonl')) as f:
        for line in f:
            rec = json.loads(line)
            steps.append(rec['step'])
            train.append(rec['train_mse'])
            if 'eval' in rec:
                es.append(rec['step'])
                ev.append(rec['eval']['mse'])
                sh.append(rec.get('eval_shuffled', {}).get('mse'))
    return steps, train, es, ev, sh


def _style(ax, title, ylabel='mse'):
    ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold')
    ax.set_xlabel('training step', color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _floor(ax, floor):
    ax.axhline(floor, color=MUTED, linestyle='--', linewidth=1.2, zorder=1)
    ax.text(0.01, floor, 'constant floor', color=MUTED, fontsize=8,
            va='bottom', ha='left', transform=ax.get_yaxis_transform())


def ramp(n):
    if n <= 1:
        return [RAMP[4]]
    return [RAMP[round(i * (len(RAMP) - 1) / (n - 1))] for i in range(n)]


def plot_run(run_dir, final=None, out=None):
    final = final or json.load(open(os.path.join(run_dir, 'final.json')))
    steps, train, es, ev, sh = read_log(run_dir)
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=130)
    _floor(ax, final['floor']['mse'])
    ax.plot(steps, train, color=RAMP[1], linewidth=1, label='train mse',
            zorder=2)
    ax.plot(es, ev, color=RAMP[6], linewidth=2, label='eval mse', zorder=3)
    if all(s is not None for s in sh):
        ax.plot(es, sh, color=RAMP[6], linewidth=1, linestyle=':',
                label='eval mse, dirs shuffled', zorder=2)
    ax.set_yscale('log')
    title = (f'{final["model"]}/{final["size"]}/{final.get("run", "")}  '
             f'{final["params"]:,} params, lr {final["args"]["lr"]:.2g}')
    _style(ax, title, 'mse (log)')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = out or os.path.join(run_dir, 'curve.png')
    fig.savefig(out)
    plt.close(fig)
    return out


def draw_model(ax, runs, title, shuffled=False, log=True):
    """Every run in ``runs`` (dicts from ``mapping.results``) on ``ax``."""
    runs = sorted(runs, key=lambda r: r['params'])
    colors = ramp(len(runs))
    _floor(ax, runs[0]['floor']['mse'])
    for r, color in zip(runs, colors):
        _, _, es, ev, sh = read_log(r['path'])
        ax.plot(es, ev, color=color, linewidth=2, zorder=3,
                label=f'{r["size"]} ({r["params"] / 1e6:.2g}M)')
        ax.text(es[-1], ev[-1], f' {r["size"]}', color=INK, fontsize=8,
                va='center')
        if shuffled and all(s is not None for s in sh):
            ax.plot(es, sh, color=color, linewidth=1, linestyle=':',
                    zorder=2)
    if log:
        ax.set_yscale('log')
    _style(ax, title, 'eval mse (log)' if log else 'eval mse')
    ax.legend(frameon=False, fontsize=8, title='size (params)',
              title_fontsize=8)


def plot_model(model, runs, out, shuffled=False):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    draw_model(ax, runs, model, shuffled)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
