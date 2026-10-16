"""Figures for the blind probe. From the repository root:

    python experiments/lift_blind_probe/figures.py

Reads runs/*/log.json, probe.json, probe.npz; writes figures/*.png and
prints the tables used in the README.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import walk                                                      # noqa: E402

D = os.path.join(os.path.dirname(__file__), 'runs')
F = os.path.join(os.path.dirname(__file__), 'figures')
MODEL_COLOR = {'gdn': '#2a78d6', 'attn': '#eb6834', 'gdn_b1': '#1baf7a', 'cdiag': '#6250d6'}
MODEL_NAME = {'gdn': 'GDN beta in (0,2)', 'gdn_b1': 'GDN beta in (0,1)', 'attn': 'attention only',
              'cdiag': 'complex-diagonal'}
N_COLOR = {4: '#a9c8ee', 8: '#78a8e3', 16: '#4f8ad8', 32: '#2a6cc0', 0: '#123f7a'}
N_NAME = {0: 'continuous', 4: 'n=4', 8: 'n=8', 16: 'n=16', 32: 'n=32'}
BINS = ['1_1', '2_4', '5_8', '9_16', '17_32']
BIN_X = [1, 3, 6.5, 12.5, 24.5]
BAND = walk.BAND_DEG
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'legend.frameon': False})


def runs():
    out = {}
    for name in sorted(os.listdir(D)):
        p = os.path.join(D, name)
        if os.path.isdir(p) and os.path.exists(f'{p}/log.json'):
            cfg = json.load(open(f'{p}/config.json'))
            out[name] = dict(cfg=cfg, log=json.load(open(f'{p}/log.json')),
                             done=os.path.exists(f'{p}/final.pt'),
                             probe=json.load(open(f'{p}/probe.json')) if os.path.exists(f'{p}/probe.json') else None,
                             npz=dict(np.load(f'{p}/probe.npz')) if os.path.exists(f'{p}/probe.npz') else None)
    return out


def breakaway(log, thr, key='pnmse_17_32'):
    for r in log:
        if r['train'][key] <= thr:
            return r['step']
    return None


def rotating_head(r):
    """The head whose fitted plane is most concentrated (largest 2nd eigenvalue
    of the mean projector), i.e. the one that keeps one rotating plane."""
    best, score = None, -1
    for tag, s in r['probe'].items():
        if 'calibration' not in s:
            continue
        sc = s['plane_spectrum'][1] if 'plane_spectrum' in s else max(s.get('pair_scores', [0]))
        if sc < 0.5:
            continue
        if sc > score:
            best, score = tag, sc
    return best


def shade(ax):
    ax.axvspan(*BAND, color='#eda100', alpha=0.18, lw=0, label='held-out band')


def calibration_panel(ax, r, tag, title, show_legend=False):
    z = r['npz']
    w = np.degrees(z[f'{tag}__w']); phi = np.degrees(z[f'{tag}__phi']); ep = z[f'{tag}__ep']
    hold = ep >= 256
    shade(ax)
    ax.plot([-180, 180], [-180, 180], color=INK2, lw=1, ls='--', label='slope 1 (homomorphism)')
    inb = (w >= BAND[0]) & (w <= BAND[1])
    ax.scatter(w[hold & ~inb], phi[hold & ~inb], s=2, color='#2a78d6', alpha=0.35, lw=0,
               label='trained angles')
    ax.scatter(w[hold & inb], phi[hold & inb], s=3, color='#e34948', alpha=0.7, lw=0,
               label='held-out angles')
    c = r['probe'][tag]['calibration']
    ax.text(0.03, 0.97, f"out of band: slope {c['out_of_band']['slope']:.3f}, resid RMS {c['out_of_band']['resid_rms']:.1f} deg\n"
            f"in band: slope {c['in_band']['slope']:.2f}, resid RMS {c['in_band']['resid_rms']:.1f} deg",
            transform=ax.transAxes, va='top', fontsize=8, color=INK)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    ax.set_title(title, fontsize=10, color=INK)
    if show_legend:
        ax.legend(loc='lower right', fontsize=7, markerscale=4)


def fig_eigen(R):
    have = [k for k, r in R.items() if r['probe'] and rotating_head(r)]
    if not have:
        return
    main = R['gdn_n0'] if 'gdn_n0' in have else R[have[0]]
    tag = rotating_head(main)
    others = [(k, R[k]) for k in ('gdn_n4', 'gdn_n8', 'gdn_n16', 'gdn_n32', 'gdn_b1_n0', 'cdiag_n0')
              if k in R and R[k]['probe']]
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0:2, 0:2])
    calibration_panel(ax, main, tag, f'GDN beta in (0,2), continuous turns, head {tag}', True)
    ax.set_xlabel('commanded turn w (deg)'); ax.set_ylabel('measured eigenangle (deg)')
    # residual panel
    axr = fig.add_subplot(gs[2, 0:2])
    z = main['npz']
    w = np.degrees(z[f'{tag}__w']); phi = np.degrees(z[f'{tag}__phi']); hold = z[f'{tag}__ep'] >= 256
    res = np.degrees(np.angle(np.exp(1j * np.radians(phi - w))))
    shade(axr)
    axr.scatter(w[hold], res[hold], s=2, color='#2a78d6', alpha=0.35, lw=0)
    edges = np.arange(-180, 181, 10)
    med = [np.median(res[hold & (w >= a) & (w < b)]) for a, b in zip(edges[:-1], edges[1:])]
    axr.plot(edges[:-1] + 5, med, color=INK, lw=1.5, label='median per 10 deg')
    axr.axhline(0, color=INK2, lw=1, ls='--')
    axr.set_xlim(-180, 180); axr.set_ylim(-30, 30); axr.set_xticks(range(-180, 181, 45))
    axr.set_xlabel('commanded turn w (deg)'); axr.set_ylabel('residual (deg)')
    axr.set_title('residual, measured minus commanded (clipped to +-30)', fontsize=10)
    axr.legend(loc='lower left', fontsize=7)
    for i, (k, r) in enumerate(others):
        a = fig.add_subplot(gs[i // 3, 2 + i % 3])
        t = rotating_head(r)
        if t is None:
            a.text(0.5, 0.5, 'no rotating plane', ha='center', transform=a.transAxes); a.set_title(k); continue
        n = r['cfg']['n']
        title = (f"GDN (0,2) trained on {N_NAME[n]}" if r['cfg']['kind'] == 'gdn'
                 else f"{MODEL_NAME[r['cfg']['kind']]}, continuous")
        calibration_panel(a, r, t, f'{title}, {t}')
        if n:
            for g in np.arange(-180, 180, 360 / n):
                a.axvline(g, color=INK2, lw=0.4, alpha=0.5)
        a.tick_params(labelsize=7)
    fig.suptitle('Held-out-angle eigenangle calibration: per-step delta-rule transition, signed angle of its rotating plane\n'
                 '(probe on continuous turns over the full circle; held-out band [50, 70] deg never trained; thin lines = training grid)',
                 fontsize=11, color=INK)
    fig.savefig(f'{F}/eigen_calibration.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_error_vs_lag(R):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    for kind in ('gdn', 'attn', 'gdn_b1', 'cdiag'):
        k = f'{kind}_n0'
        if k not in R:
            continue
        last = R[k]['log'][-1]['train']
        axs[0, 0].plot(BIN_X, [last[f'pos_{b}'] for b in BINS], marker='o', color=MODEL_COLOR[kind], lw=2, label=MODEL_NAME[kind])
        axs[0, 1].plot(BIN_X, [last[f'head_{b}'] for b in BINS], marker='o', color=MODEL_COLOR[kind], lw=2, label=MODEL_NAME[kind])
    for n in (4, 8, 16, 32, 0):
        k = f'gdn_n{n}'
        if k not in R:
            continue
        last = R[k]['log'][-1]['train']
        axs[1, 0].plot(BIN_X, [last[f'pos_{b}'] for b in BINS], marker='o', color=N_COLOR[n], lw=2, label=N_NAME[n])
        axs[1, 1].plot(BIN_X, [last[f'head_{b}'] for b in BINS], marker='o', color=N_COLOR[n], lw=2, label=N_NAME[n])
    for a in axs.ravel():
        a.set_xscale('log', base=2); a.set_xticks(BIN_X); a.set_xticklabels(['1', '2-4', '5-8', '9-16', '17-32'])
        a.set_xlabel('composition length (lag t - s, steps)')
        a.legend(fontsize=8)
    for a in axs[:, 0]:
        a.set_ylabel('position RMS error (walk units)')
    for a in axs[:, 1]:
        a.set_ylabel('median heading error (deg)'); a.set_yscale('log')
    axs[0, 0].set_title('models, continuous turns (training distribution)')
    axs[0, 1].set_title('models, continuous turns (training distribution)')
    axs[1, 0].set_title('GDN beta in (0,2) by granularity (own training distribution)')
    axs[1, 1].set_title('GDN beta in (0,2) by granularity (own training distribution)')
    fig.suptitle('Pose error vs composition length at the end of training (100k steps). '
                 'Typical displacement at lag L is ~0.89 sqrt(L) units.', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{F}/error_vs_lag.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_heldout(R):
    keys = [k for k in ('gdn_n0', 'attn_n0', 'gdn_b1_n0', 'cdiag_n0', 'gdn_n4', 'gdn_n32') if k in R]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for a, k in zip(axs, keys):
        last = R[k]['log'][-1]
        a.plot(BIN_X, [last['ctrl'][f'head_{b}'] for b in BINS], marker='o', color='#2a78d6', lw=2, label='turns in trained [20,40]')
        a.plot(BIN_X, [last['band'][f'head_{b}'] for b in BINS], marker='s', color='#e34948', lw=2, label='turns in held-out [50,70]')
        a.set_xscale('log', base=2); a.set_xticks(BIN_X); a.set_xticklabels(['1', '2-4', '5-8', '9-16', '17-32'], fontsize=7)
        a.set_yscale('log'); a.set_title(k, fontsize=10); a.set_xlabel('lag (steps)')
    axs[0].set_ylabel('median heading error (deg)')
    axs[0].legend(fontsize=7)
    fig.suptitle('Walks whose every turn lies in one 20-deg band: held-out band vs a trained control band', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{F}/heldout_walks.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_beta(R):
    keys = [k for k in ('gdn_n0', 'gdn_b1_n0', 'gdn_n4') if k in R and R[k]['npz'] is not None]
    if not keys:
        return
    fig, axs = plt.subplots(len(keys), 2, figsize=(11, 3.3 * len(keys)), squeeze=False)
    for row, k in zip(axs, keys):
        z = R[k]['npz']
        for a, head in zip(row, ('l2r0', 'r2l0')):
            w = np.degrees(z[f'{head}__w'])
            shade(a)
            for tok, col in (('odo', '#2a78d6'), ('anchor', '#eb6834')):
                b = z[f'{head}__beta_{tok}']
                a.scatter(w, b, s=1.5, color=col, alpha=0.25, lw=0)
                edges = np.arange(-180, 181, 10)
                med = [np.median(b[(w >= lo) & (w < hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
                a.plot(edges[:-1] + 5, med, color=col, lw=2, label=f'{tok} token (median per 10 deg)')
            a.set_xlim(-180, 180); a.set_xticks(range(-180, 181, 45))
            a.set_ylim(0, 2.05 if 'b1' not in k else 1.05)
            a.set_title(f'{k}: {"left-to-right" if head == "l2r0" else "right-to-left"} head', fontsize=10)
            a.set_xlabel('turn w of the step (deg)'); a.set_ylabel('beta')
            a.legend(fontsize=7, loc='lower left')
    fig.suptitle('Learned write strength beta by turn angle (probe walks, continuous turns)', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{F}/beta_by_turn.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def load_chart(r, name):
    p = os.path.join(D, name, 'chart.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z['grid'], np.degrees(z['step16']), json.load(open(os.path.join(D, name, 'chart.json')))


def chart_panel(ax, name, R, title, big=False):
    c = load_chart(R[name], name)
    if c is None:
        return False
    gd, ch, js = c
    y = ch[:, :, 0]                                              # (grid, base): j = i
    shade(ax)
    ax.plot([-180, 180], [-180, 180], color=INK2, lw=1, ls='--', label='slope 1 (homomorphism)')
    lo, hi = np.percentile(y, [10, 90], axis=1)
    ax.fill_between(gd, lo, hi, color='#2a78d6', alpha=0.2, lw=0, label='10-90% over base walks')
    ax.plot(gd, y.mean(1), color='#2a78d6', lw=2, label='mean chart')
    n = R[name]['cfg']['n']
    if n:
        for g in np.arange(-180, 180, 360 / n):
            ax.axvline(g, color=INK2, lw=0.4, alpha=0.6)
    s = js['step16']['same_step']
    ax.text(0.03, 0.97, f"out of band: slope {s['out_of_band']['slope']:.2f}, RMS vs identity {s['out_of_band']['resid_rms_vs_identity']:.1f} deg\n"
            f"in band: slope {s['in_band']['slope']:.2f}, RMS vs identity {s['in_band']['resid_rms_vs_identity']:.1f} deg\n"
            f"decoder R2 (unwrapped heading) {js['decoder']['r2_unwrapped']:.2f}",
            transform=ax.transAxes, va='top', fontsize=8 if big else 6.5, color=INK)
    ax.set_xlim(-180, 180); ax.set_ylim(-200, 200)
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    ax.set_title(title, fontsize=10 if big else 9)
    if big:
        ax.legend(loc='lower right', fontsize=7)
    return True


def fig_chart(R):
    if 'gdn_n0' not in R or load_chart(R['gdn_n0'], 'gdn_n0') is None:
        return
    others = [k for k in ('gdn_n4', 'gdn_n8', 'gdn_n16', 'gdn_n32', 'attn_n0', 'gdn_b1_n0', 'cdiag_n0') if k in R]
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(3, 6, hspace=0.6, wspace=0.4)
    ax = fig.add_subplot(gs[0:2, 0:2])
    chart_panel(ax, 'gdn_n0', R, 'GDN beta in (0,2), trained on continuous turns', big=True)
    ax.set_xlabel('commanded turn w\' at step 16 (deg)'); ax.set_ylabel('change in decoded unwrapped heading (deg)')
    gd, ch, _ = load_chart(R['gdn_n0'], 'gdn_n0')
    axr = fig.add_subplot(gs[2, 0:2])
    shade(axr)
    axr.plot(gd, ch[:, :, 0].mean(1) - gd, color='#2a78d6', lw=2, label='mean chart minus identity')
    axr.axhline(0, color=INK2, lw=1, ls='--')
    axr.set_xlim(-180, 180); axr.set_xticks(range(-180, 181, 45)); axr.set_ylim(-25, 25)
    axr.set_xlabel('commanded turn w\' (deg)'); axr.set_ylabel('residual (deg)')
    axr.set_title('residual of the mean chart', fontsize=10); axr.legend(fontsize=7, loc='lower left')
    for i, k in enumerate(others):
        a = fig.add_subplot(gs[i // 4, 2 + i % 4])
        n = R[k]['cfg']['n']
        t = (f"GDN (0,2) trained on {N_NAME[n]}" if R[k]['cfg']['kind'] == 'gdn'
             else f"{MODEL_NAME[R[k]['cfg']['kind']]}, continuous")
        if not chart_panel(a, k, R, t):
            a.set_visible(False)
        a.tick_params(labelsize=7)
    fig.suptitle('Held-out-angle calibration of the learned angle chart: change one step\'s turn, read the linearly decoded '
                 'unwrapped heading after block 0\n(decoder fit on each model\'s own training distribution, band excluded; '
                 'shaded = held-out band [50, 70] deg; thin lines = training grid)', fontsize=11)
    fig.savefig(f'{F}/calibration.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_heldout_mirror(R):
    keys = [k for k in ('gdn_n0', 'attn_n0', 'gdn_b1_n0', 'cdiag_n0', 'gdn_n4', 'gdn_n32')
            if k in R and os.path.exists(os.path.join(D, k, 'heldout.json'))]
    if not keys:
        return
    fig, axs = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 6.4), sharey='row', squeeze=False)
    for j, k in enumerate(keys):
        h = json.load(open(os.path.join(D, k, 'heldout.json')))
        for i, mode in enumerate(('mix', 'const')):
            a = axs[i, j]
            a.plot(BIN_X, [h[f'{mode}_mirror'][f'head_{b}'] for b in BINS], marker='o', color='#2a78d6', lw=2,
                   label='trained mirror [-70,-50]')
            a.plot(BIN_X, [h[f'{mode}_band'][f'head_{b}'] for b in BINS], marker='s', color='#e34948', lw=2,
                   ls='--', label='held-out [50,70]')
            a.set_xscale('log', base=2); a.set_xticks(BIN_X)
            a.set_xticklabels(['1', '2-4', '5-8', '9-16', '17-32'], fontsize=7)
            a.set_yscale('log'); a.set_xlabel('lag (steps)')
            a.set_title(f"{k}: {'25% band turns' if mode == 'mix' else 'all turns in band'}", fontsize=9)
    axs[0, 0].set_ylabel('median heading error (deg)'); axs[1, 0].set_ylabel('median heading error (deg)')
    axs[0, 0].legend(fontsize=7)
    fig.suptitle('Held-out turns vs their trained mirror image (identical walk statistics by symmetry)', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{F}/heldout_mirror.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


THRS = (0.5, 0.1, 0.02)


def fig_breakaway(R):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for n in (4, 8, 16, 32, 0):
        k = f'gdn_n{n}'
        if k not in R:
            continue
        L = R[k]['log']
        axs[0].plot([r['step'] for r in L], [r['train']['pnmse_17_32'] for r in L], color=N_COLOR[n], lw=1.8, label=N_NAME[n])
    for kind in ('attn', 'gdn_b1', 'cdiag'):
        k = f'{kind}_n0'
        if k in R:
            L = R[k]['log']
            axs[0].plot([r['step'] for r in L], [r['train']['pnmse_17_32'] for r in L], color=MODEL_COLOR[kind],
                        lw=1.2, ls=':', label=f'{MODEL_NAME[kind]}, continuous')
    for t in THRS:
        axs[0].axhline(t, color=INK2, lw=0.8, ls='--')
    axs[0].set_yscale('log'); axs[0].set_xlabel('training step'); axs[0].set_ylabel('position nmse, lags 17-32')
    axs[0].set_title('Long-lag position error during training (dashed: breakaway thresholds)', fontsize=10)
    axs[0].legend(fontsize=7)
    xs = [4, 8, 16, 32, 64]
    for t, mk in zip(THRS, ('o', 's', '^')):
        ys, cens = [], []
        for n in (4, 8, 16, 32, 0):
            k = f'gdn_n{n}'
            b = breakaway(R[k]['log'], t) if k in R and R[k]['done'] else None
            ys.append(b if b is not None else np.nan)
            cens.append(k in R and R[k]['done'] and b is None)
        axs[1].plot(xs, ys, marker=mk, lw=1.8, color=INK2 if t != 0.1 else '#2a78d6', label=f'threshold {t}')
        for x, c in zip(xs, cens):
            if c:
                axs[1].scatter([x], [100000], marker='^', facecolor='none', edgecolor=INK, s=50)
    axs[1].set_xscale('log', base=2); axs[1].set_xticks(xs); axs[1].set_xticklabels(['4', '8', '16', '32', 'cont.'])
    axs[1].set_xlabel('turn granularity n'); axs[1].set_ylabel('breakaway step')
    axs[1].set_ylim(0, 105000)
    axs[1].set_title('Breakaway step vs granularity (GDN beta in (0,2); open triangle = not reached by 100k)', fontsize=10)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{F}/breakaway.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def tables(R):
    print('| run | done | pos 1 | pos 5-8 | pos 17-32 | head 1 | head 17-32 | pnmse 17-32 | cont nmse | band head 17-32 | ctrl head 17-32 | breakaway 0.5 / 0.1 / 0.02 |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for k, r in R.items():
        L = r['log'][-1]
        tr = L['train']
        br = ' / '.join(str(breakaway(r['log'], t) or '>' + str(L['step'])) for t in THRS)
        print(f"| {k} | {r['done']} | {tr['pos_1_1']:.3f} | {tr['pos_5_8']:.3f} | {tr['pos_17_32']:.3f} | {tr['head_1_1']:.2f} | "
              f"{tr['head_17_32']:.2f} | {tr['pnmse_17_32']:.4f} | {L['cont']['nmse']:.4f} | {L['band']['head_17_32']:.1f} | "
              f"{L['ctrl']['head_17_32']:.1f} | {br} |")
    print()
    for k, r in R.items():
        if not r['probe']:
            continue
        for tag, s in r['probe'].items():
            if 'calibration' not in s:
                continue
            c = s['calibration']
            print(f"{k} {tag}: spectrum {s.get('plane_spectrum', '')[:3] if 'plane_spectrum' in s else s.get('pair')}  "
                  f"out slope {c['out_of_band']['slope']:.3f} rms {c['out_of_band']['resid_rms']:.2f} med {c['out_of_band']['resid_median_abs']:.2f} "
                  f"w10 {c['out_of_band']['frac_within_10']:.3f} | in slope {c['in_band']['slope']:.3f} rms {c['in_band']['resid_rms']:.2f} "
                  f"mean {c['in_band']['resid_mean']:.2f} w10 {c['in_band']['frac_within_10']:.3f}  "
                  f"inv {s.get('invariance_residual_median', float('nan')):.3f} mod {s.get('modulus_median', float('nan')):.3f} "
                  f"comp {s.get('composition_within_10deg', '')}")


if __name__ == '__main__':
    os.makedirs(F, exist_ok=True)
    R = runs()
    fig_chart(R); fig_eigen(R); fig_error_vs_lag(R); fig_heldout(R); fig_heldout_mirror(R); fig_beta(R); fig_breakaway(R)
    tables(R)
