"""Is the delta-rule state transition over one walk step a rotation that
tracks the turn?

    python tools/transition_probe.py results/se2/hybrid4/xs/seed0_100k_qt/final.pt

For a trained SE(2) hybrid4-family checkpoint whose first block is the
Gated DeltaNet layer, recomputes that layer's per-token keys, decays and
write strengths exactly as the layer does (on CPU), then, for every walk
step, multiplies the homogeneous part of the state update

    A_tok = a_tok * (I - beta_tok k_tok k_tok^T)        (32 x 32, key space)

over the step's span in each head's own scan order:
  left-to-right head:  odometry token t, then the 9 patches of observation t+1
  right-to-left head:  odometry token t, then everything up to and including
                       the 9 patches of observation t (queries asked at t
                       sit in between and are included)
The eigenvalues of the product are basis-free. If the head carries a
quantity that rotates with the agent, steps with a quarter turn should
produce eigenvalue pairs at +-90 degrees, turning around eigenvalues at 180
degrees, and no turn eigenvalues near 0 degrees. Reports, per turn class,
how the angles of the significant eigenvalues (|lambda| > 0.5) are
distributed, and the median number of significant eigenvalues.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402


class Stop(Exception):
    pass


def layer_quantities(gdn, x):
    """k (unit), g, beta for every token, in each head's scan order, exactly
    as GatedDeltaNet.forward computes them. Returns arrays (B, L, H, d),
    (B, L, H), (B, L, H) where the right-to-left heads are in reversed
    sequence order."""
    mod = sys.modules[type(gdn).__module__]
    b, l, _ = x.shape
    h, d, half = gdn.heads, gdn.d, gdn.heads // 2
    qkv = torch.stack([gdn.q(x), gdn.k(x), gdn.v(x)], 2).view(b, l, 3, h, d)
    g = F.logsigmoid(gdn.decay(x).float()) / mod.DECAY_TAU
    src = open(mod.__file__).read()
    beta = (torch.sigmoid(gdn.beta(x).float()) if 'beta = torch.sigmoid' in src
            else 2 * torch.sigmoid(gdn.beta(x).float()))
    fwd, bwd = qkv[:, :, :, :half], qkv[:, :, :, half:].flip(1)
    seq = torch.cat([fwd, bwd], 3).reshape(b, l, 3 * h * d)
    if 'ABLATION: no short conv' in src:
        seq = F.silu(seq).view(b, l, 3, h, d)
    else:
        seq = F.pad(seq.transpose(1, 2), (gdn.conv - 1, 0))
        seq = F.silu(gdn.short(seq)).transpose(1, 2).view(b, l, 3, h, d)
    k = F.normalize(seq[:, :, 1].float(), dim=-1)
    gh = torch.cat([g[:, :, :half], g[:, :, half:].flip(1)], 2)
    bh = torch.cat([beta[:, :, :half], beta[:, :, half:].flip(1)], 2)
    return k.numpy(), gh.numpy(), bh.numpy()


def main(path, batches=4):
    model, args, spec, task = load_checkpoint(path, 'cpu')
    t = tasks.get(task)
    config = tasks.config_from_args(task, spec, args)
    _, test_bank = load_banks(args['bank'], config)
    data = eval_set(config, test_bank, 64, batches, 4242)
    gdn = model.blocks[0].mixer
    assert type(gdn).__name__ == 'GatedDeltaNet'
    cap = {}

    def hook(mod, inp):
        cap['x'] = inp[0].detach()
        raise Stop
    hk = gdn.register_forward_pre_hook(hook)
    H = gdn.heads; half = H // 2
    angles = {(d, tn): [] for d in ('l2r', 'r2l') for tn in range(4)}
    counts = {(d, tn): [] for d in ('l2r', 'r2l') for tn in range(4)}
    inband = {(d, tn): [] for d in ('l2r', 'r2l') for tn in range(4)}
    for bt in data:
        obs, odo, pose, tgt, qt = t['task'].prepare(bt, 'cpu')
        try:
            model(obs, odo, pose, query_time=qt)
        except Stop:
            pass
        with torch.no_grad():
            k, g, beta = layer_quantities(gdn, cap['x'])
        B, L = k.shape[:2]
        t1, nq = obs.shape[1], qt.shape[1]
        # labels in layout order: ('o', step, patch) / ('d', step) / ('q', ...)
        layout = []
        for s in range(t1):
            layout += [('o', s)] * 9 + ([('d', s)] if s < t1 - 1 else [])
        order = model.token_order(qt, t1, nq).numpy()
        for i in range(B):
            qlab = [('q', int(qt[i, j])) for j in range(nq) for _ in range(9)]
            seq = [(layout + qlab)[o] for o in order[i]]
            for d, heads, lab in (('l2r', range(half), seq), ('r2l', range(half, H), seq[::-1])):
                dpos = [p for p, x in enumerate(lab) if x[0] == 'd']
                for p in dpos:
                    s = lab[p][1]
                    target = s + 1 if d == 'l2r' else s       # observation whose patches close the span
                    end = p + 1
                    seen = 0
                    while end < L and seen < 9:
                        if lab[end] == ('o', target):
                            seen += 1
                        end += 1
                    turn = int(odo[i, s]) // 4
                    for hh in heads:
                        P = np.eye(k.shape[-1])
                        for j in range(p, end):
                            kj = k[i, j, hh][:, None]
                            A = np.exp(g[i, j, hh]) * (np.eye(len(kj)) - beta[i, j, hh] * (kj @ kj.T))
                            P = A @ P
                        ev = np.linalg.eigvals(P)
                        sig = ev[np.abs(ev) > 0.5]
                        angles[(d, turn)] += np.degrees(np.abs(np.angle(sig))).tolist()
                        counts[(d, turn)].append(len(sig))
                        band = 90 if turn in (1, 3) else (180 if turn == 2 else None)
                        if band is not None:
                            a = np.degrees(np.abs(np.angle(ev)))
                            m = (np.abs(a - band) < 15) & (np.abs(ev) > 0.5)
                            inband[(d, turn)].append((m.sum(), a[m].tolist(), np.abs(ev[m]).tolist()))
    hk.remove()
    names = ['none', 'right (90)', 'around (180)', 'left (90)']
    print(path)
    for d in ('l2r', 'r2l'):
        print(f'  {"left-to-right" if d == "l2r" else "right-to-left"} head(s): angles of eigenvalues with |lambda| > 0.5, per step')
        print(f'    {"turn":14s} {"steps":>6s} {"sig. eigs":>9s}   {"near 0":>7s} {"near 90":>7s} {"near 180":>8s} {"other":>6s}')
        for tn in range(4):
            a = np.array(angles[(d, tn)]); c = counts[(d, tn)]
            if not len(c) or not len(a):
                print(f'    {names[tn]:14s} no eigenvalue with |lambda| > 0.5 (the step contracts the state)')
                continue
            n0 = (a < 15).mean(); n90 = (np.abs(a - 90) < 15).mean(); n180 = (a > 165).mean()
            print(f'    {names[tn]:14s} {len(c) // max(1, (half)):6d} {np.median(c):9.0f}   '
                  f'{100 * n0:6.1f}% {100 * n90:6.1f}% {100 * n180:7.1f}% {100 * (1 - n0 - n90 - n180):5.1f}%')
            if inband[(d, tn)]:
                nb = np.array([x[0] for x in inband[(d, tn)]])
                ang = np.concatenate([x[1] for x in inband[(d, tn)]] or [[np.nan]])
                mod = np.concatenate([x[2] for x in inband[(d, tn)]] or [[np.nan]])
                vals, cnt = np.unique(nb, return_counts=True)
                dist = ', '.join(f'{v}: {100 * c / len(nb):.1f}%' for v, c in zip(vals, cnt))
                print(f'      eigenvalues in the expected band, per step: {dist}; '
                      f'angle median {np.nanmedian(ang):.1f} deg (10-90%: {np.nanpercentile(ang, 10):.1f}-{np.nanpercentile(ang, 90):.1f}), '
                      f'|lambda| median {np.nanmedian(mod):.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
