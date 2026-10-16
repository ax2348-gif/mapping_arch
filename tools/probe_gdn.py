"""read_beta and transition_probe, per head, for any Gated DeltaNet layer of an
SE(2) checkpoint (hybrid4 family, hybrid4.5b coprocessors at any placement,
hybrid4_convrace's per-head conv widths), from final.pt or a step<k>.pt.

    python tools/probe_gdn.py results/se2/hybrid4_convrace/xs/seed0/final.pt
    python tools/probe_gdn.py <ckpt> --layer 1 --json out.json

The layer's input is captured with a forward pre-hook (the forward stops
there), then k, the log decay and beta are recomputed per token exactly as
the layer computes them: the layer's own ``scan_inputs`` when it has one
(hybrid4_convrace), else hybrid4's GatedDeltaNet computation (as in
tools/transition_probe.py). On hybrid4 xs it reproduces read_beta.py and
transition_probe.py (checked in docs/lift/prefreeze_ablations.md).

Per head it reports:
- beta by token role (observation patches, queries, odometry by turn), as
  read_beta.py, plus the patch profile: median beta of the j-th patch after
  an odometry token in the head's scan order (j = 1..9, steps with no query
  in between). beta is computed from the token's own input (it does not
  pass through the convolution), so a patch's beta does not depend on the
  turn; what the profile shows is which patch positions the head gives the
  reflection role. Their keys carry the turn only through the convolution,
  so reflection-role patches inside the conv's reach are the second mirror.
- crispness = P(beta >= 1.9 | odometry token) - P(beta >= 1.9 | patch): how
  cleanly the reflection role is assigned to odometry tokens (a definition
  of this tool, not of the hypothesis).
- the transition probe: for every walk step, the product of
  ``a (I - beta k k^T)`` over the step's span in the head's scan order
  (transition_probe.py's spans), its eigenvalues with |lambda| > 0.5, and
  the fractions of quarter-turn steps with a pair within 15 degrees of 90,
  of around steps with a pair within 15 degrees of 180, and of no-turn steps
  with any eigenvalue in either band (false positives).
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402

TURNS = ['none', 'right', 'around', 'left']


class Stop(Exception):
    pass


def load_any(path):
    """(model, args, spec, task) from final.pt or a step<k>.pt."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    args = state['args']
    task = state.get('task', args.get('task', 'z2'))
    t = tasks.get(task)
    if 'spec' in state:
        spec = t['task'].TaskSpec(**state['spec'])
    else:
        _, spec = tasks.config_and_spec(
            task, args['episode_steps'], args['queries'], args['seed'],
            args.get('query_times', 'last'), args.get('step_px', 9),
            args.get('odometry_dropout', 0.0), args.get('texture', 'mnist'))
    model = t['models'].get(args['model']).build(args['size'], spec)
    model.load_state_dict(state['model'])
    return model.eval(), args, spec, task


def gdn_layers(model):
    if hasattr(model, 'gdn_layers'):
        return model.gdn_layers()
    return [(f'blocks.{i}.mixer', b.mixer) for i, b in enumerate(model.blocks)
            if type(b.mixer).__name__ == 'GatedDeltaNet']


def quantities(gdn, x):
    """k (unit), g, beta per token, each head in its scan order."""
    if hasattr(gdn, 'scan_inputs'):
        _, k, _, g, beta = gdn.scan_inputs(x)
        return k.float().numpy(), g.float().numpy(), beta.float().numpy()
    mod = sys.modules[type(gdn).__module__]
    src = open(mod.__file__).read()
    assert 'ABLATION' not in src, 'ablated layers: use the original tools'
    b, l, _ = x.shape
    h, d, half = gdn.heads, gdn.d, gdn.heads // 2
    qkv = torch.stack([gdn.q(x), gdn.k(x), gdn.v(x)], 2).view(b, l, 3, h, d)
    g = F.logsigmoid(gdn.decay(x).float()) / mod.DECAY_TAU
    beta = 2 * torch.sigmoid(gdn.beta(x).float())
    fwd, bwd = qkv[:, :, :, :half], qkv[:, :, :, half:].flip(1)
    seq = torch.cat([fwd, bwd], 3).reshape(b, l, 3 * h * d)
    seq = F.pad(seq.transpose(1, 2), (gdn.conv - 1, 0))
    seq = F.silu(gdn.short(seq)).transpose(1, 2).view(b, l, 3, h, d)
    k = F.normalize(seq[:, :, 1].float(), dim=-1)
    gh = torch.cat([g[:, :, :half], g[:, :, half:].flip(1)], 2)
    bh = torch.cat([beta[:, :, :half], beta[:, :, half:].flip(1)], 2)
    return k.numpy(), gh.numpy(), bh.numpy()


def probe(path, layer=0, batches=4, seed=4242):
    model, args, spec, task = load_any(path)
    assert task == 'se2', 'SE(2) checkpoints only'
    t = tasks.get(task)
    config = tasks.config_from_args(task, spec, args)
    _, test_bank = load_banks(args['bank'], config)
    data = eval_set(config, test_bank, 64, batches, seed)
    name, gdn = gdn_layers(model)[layer]
    H, half = gdn.heads, gdn.heads // 2
    cap = {}

    def hook(mod, inp):
        cap['x'] = inp[0].detach()
        raise Stop
    hk = gdn.register_forward_pre_hook(hook)
    beta_by = {h: {} for h in range(H)}          # head -> group -> list
    prof = {h: {j: [] for j in range(1, 10)} for h in range(H)}
    eig = {h: {tn: [] for tn in range(4)} for h in range(H)}
    for bt in data:
        obs, odo, pose, tgt, qt = t['task'].prepare(bt, 'cpu')
        try:
            with torch.no_grad():
                model(obs, odo, pose, query_time=qt)
        except Stop:
            pass
        with torch.no_grad():
            k, g, beta = quantities(gdn, cap['x'])
        B, L = k.shape[:2]
        t1, nq = obs.shape[1], qt.shape[1]
        layout = []
        for s in range(t1):
            layout += [('o', s)] * 9 + ([('d', s)] if s < t1 - 1 else [])
        order = model.token_order(qt, t1, nq).numpy()
        odo_np = odo.numpy()
        for i in range(B):
            qlab = [('q', int(qt[i, j])) for j in range(nq) for _ in range(9)]
            seq = [(layout + qlab)[o] for o in order[i]]
            for heads, lab in ((range(half), seq), (range(half, H), seq[::-1])):
                heads = list(heads)
                # beta by role, in this head's scan order
                kinds = np.array([x[0] for x in lab])
                for hh in heads:
                    bb = beta[i, :, hh]
                    d = beta_by[hh]
                    d.setdefault('patch', []).extend(bb[kinds == 'o'])
                    d.setdefault('query', []).extend(bb[kinds == 'q'])
                    for p in np.nonzero(kinds == 'd')[0]:
                        tn = int(odo_np[i, lab[p][1]]) // 4
                        d.setdefault(f'odo_{TURNS[tn]}', []).append(bb[p])
                        d.setdefault('odo', []).append(bb[p])
                # spans and patch profile
                l2r = heads[0] < half
                for p in np.nonzero(kinds == 'd')[0]:
                    s = lab[p][1]
                    target = s + 1 if l2r else s
                    end, seen, offs = p + 1, 0, []
                    while end < L and seen < 9:
                        if lab[end] == ('o', target):
                            seen += 1
                            offs.append(end)
                        end += 1
                    tn = int(odo_np[i, s]) // 4
                    clean = (end - p - 1) == 9            # no query in between
                    if clean:
                        for j, pos in enumerate(offs, 1):
                            for hh in heads:
                                prof[hh][j].append(beta[i, pos, hh])
                    kk = k[i, p:end][:, heads]                       # (n, h, d)
                    A = (np.exp(g[i, p:end][:, heads])[..., None, None]
                         * (np.eye(kk.shape[-1])
                            - beta[i, p:end][:, heads][..., None, None]
                            * kk[..., :, None] * kk[..., None, :]))
                    P = np.broadcast_to(np.eye(kk.shape[-1]), A.shape[1:]).copy()
                    for j in range(A.shape[0]):
                        P = A[j] @ P
                    ev = np.linalg.eigvals(P)                        # (h, d)
                    for c, hh in enumerate(heads):
                        e = ev[c]
                        sig = np.abs(e) > 0.5
                        ang = np.degrees(np.abs(np.angle(e)))
                        eig[hh][tn].append((int((sig & (np.abs(ang - 90) < 15)).sum()),
                                            int((sig & (ang > 165)).sum()),
                                            int(sig.sum())))
    hk.remove()
    out = {'path': path, 'layer': name, 'heads': H,
           'widths': getattr(gdn, 'head_widths', [gdn.conv] * H), 'head': []}
    for hh in range(H):
        d = {g_: np.array(v) for g_, v in beta_by[hh].items()}
        r = {'head': hh, 'direction': 'l2r' if hh < half else 'r2l',
             'conv_width': out['widths'][hh], 'beta': {}}
        for g_, v in d.items():
            r['beta'][g_] = {'median': float(np.median(v)),
                             'p10': float(np.percentile(v, 10)),
                             'p90': float(np.percentile(v, 90)),
                             'gt1': float((v > 1).mean()),
                             'ge1.9': float((v >= 1.9).mean())}
        r['crispness'] = r['beta']['odo']['ge1.9'] - r['beta']['patch']['ge1.9']
        r['profile'] = [float(np.median(prof[hh][j])) for j in range(1, 10)]
        e = {tn: np.array(v) for tn, v in eig[hh].items()}
        quarter = np.concatenate([e[1], e[3]])
        r['rot90'] = float((quarter[:, 0] >= 2).mean())
        r['rot180'] = float((e[2][:, 1] >= 2).mean())
        r['false_pos'] = float(((e[0][:, 0] + e[0][:, 1]) > 0).mean())
        r['sig_eigs_median'] = float(np.median(np.concatenate([x[:, 2] for x in e.values()])))
        r['steps'] = {TURNS[tn]: int(len(v)) for tn, v in e.items()}
        out['head'].append(r)
    return out


def report(out):
    print(f'{out["path"]}  layer {out["layer"]}, {out["heads"]} heads')
    for r in out['head']:
        b = r['beta']
        print(f'  head {r["head"]} ({r["direction"]}, conv {r["conv_width"]}): '
              f'rot90 {100 * r["rot90"]:5.1f}%  rot180 {100 * r["rot180"]:5.1f}%  '
              f'no-turn false pos {100 * r["false_pos"]:4.1f}%  '
              f'sig. eigs {r["sig_eigs_median"]:.0f}  crispness {r["crispness"]:+.2f}')
        print('    beta median: ' + '  '.join(
            f'{g_} {b[g_]["median"]:.2f}' for g_ in
            ['patch', 'query', 'odo_none', 'odo_right', 'odo_around', 'odo_left']
            if g_ in b))
        print('    beta of patch j = 1..9 after the odometry token: '
              + ' '.join(f'{v:.2f}' for v in r['profile']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt', nargs='+')
    p.add_argument('--layer', type=int, default=0,
                   help='index among the Gated DeltaNet layers, forward order')
    p.add_argument('--batches', type=int, default=4)
    p.add_argument('--json', default=None, help='write all results here')
    a = p.parse_args()
    outs = []
    for path in a.ckpt:
        o = probe(path, a.layer, a.batches)
        report(o)
        outs.append(o)
    if a.json:
        json.dump(outs, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
