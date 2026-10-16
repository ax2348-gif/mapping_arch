"""Is the time-only scan's per-step transition a rotation by the turn?

    python tools/time_scan_probe.py results/se2d/hybrid4_time/xs/seed0_100k_qt/step50000.pt

For an se2d ``hybrid4_time`` / ``hybrid4_time_full`` checkpoint: hooks the
time tokens entering each time-scan block, recomputes that block's keys,
betas and decays exactly as the layer does (``TimeScan.quantities``), and
for every odometry token and head forms the per-step transition in key
space

    A_t = a_t (I - b2 k2 k2^T)(I - b1 k1 k1^T)          (DeltaProduct, n = 2)

(the same matrix in both scan directions: the two Householder steps keep
their order inside a token). Reports, per turn class of the step (none,
right 90, around 180, left 90), the angles of the eigenvalues with
|lambda| > 0.5, the fraction of steps with an eigenvalue within 10 deg of
the turn's angle, beta per Householder step, and det A_t / a_t^d.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402
from mapping.models.se2d.hybrid4_time import TimeBlock  # noqa: E402

NAMES = ['none (0)', 'right (90)', 'around (180)', 'left (90)']
WANT = [0, 90, 180, 90]


def main(path, batches=4):
    model, args, spec, task = load_checkpoint(path, 'cuda')
    config = tasks.config_from_args(task, spec, args)
    _, test_bank = load_banks(args['bank'], config, args.get('texture', 'mnist'))
    data = eval_set(config, test_bank, 64, batches, 4242)
    t = tasks.get(task)
    blocks = [(i, b) for i, b in enumerate(model.blocks) if isinstance(b, TimeBlock)]
    cap = {}
    hooks = [b.mixer.register_forward_pre_hook(
        lambda mod, a, i=i: cap.__setitem__(i, (a[0].detach(), a[1]['write'], a[1]['tstep'])))
        for i, b in blocks]
    print(path)
    stats = {i: {tn: dict(ang=[], hit=[], beta=[], det=[], head=[]) for tn in range(4)} for i, _ in blocks}
    for bt in data:
        obs, odo, pose, tgt, qt = t['task'].prepare(bt, 'cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            model(obs, odo, pose, query_time=qt)
        for i, b in blocks:
            x, write, tstep = cap[i]
            with torch.no_grad():
                q, k, v, g, beta = b.mixer.quantities(x.float(), write)
            k, g, beta = k.float().cpu().numpy(), g.float().cpu().numpy(), beta.float().cpu().numpy()
            wr, st = write.cpu().numpy(), tstep.cpu().numpy()
            turns = (odo.cpu().numpy() // 4)                          # (B, 32)
            B, L, n, H, d = k.shape
            for e in range(B):
                for p in np.nonzero(wr[e])[0]:
                    tn = int(turns[e, st[e, p]])
                    for hh in range(H):
                        A = np.eye(d)
                        for j in range(n):
                            kj = k[e, p, j, hh][:, None]
                            A = (np.eye(d) - beta[e, p, j, hh] * kj @ kj.T) @ A
                        ev = np.linalg.eigvals(A * np.exp(g[e, p, hh]))
                        sig = np.degrees(np.abs(np.angle(ev[np.abs(ev) > 0.5])))
                        s_ = stats[i][tn]
                        s_['ang'] += sig.tolist()
                        s_['hit'].append(bool(np.any(np.abs(sig - WANT[tn]) < 10)) if tn else
                                         bool(len(sig) and np.all(sig < 10)))
                        s_['beta'] += beta[e, p, :, hh].tolist()
                        s_['det'].append(np.linalg.det(A))
                        s_['head'].append(hh)
    for h in hooks:
        h.remove()
    H = blocks[0][1].mixer.heads
    for i, _ in blocks:
        print(f'  time-scan block {i}: per odometry token and head')
        print(f'    {"turn":14s} {"n":>6s} {"hit":>7s}  {"near0":>6s} {"near90":>6s} {"near180":>7s}  '
              f'{"beta med":>8s} {"det>0":>6s}')
        for tn in range(4):
            s_ = stats[i][tn]
            a = np.array(s_['ang'])
            if not len(s_['hit']):
                continue
            n0, n90, n180 = ((a < 15).mean(), (np.abs(a - 90) < 15).mean(), (a > 165).mean()) if len(a) else (0, 0, 0)
            print(f'    {NAMES[tn]:14s} {len(s_["hit"]):6d} {100*np.mean(s_["hit"]):6.1f}%  '
                  f'{100*n0:5.1f}% {100*n90:5.1f}% {100*n180:6.1f}%  {np.median(s_["beta"]):8.3f} '
                  f'{100*np.mean(np.array(s_["det"]) > 0):5.1f}%   per head hit: '
                  + ' '.join(f'h{hh}({"l2r" if hh < H // 2 else "r2l"}) '
                             f'{100*np.mean(np.array(s_["hit"])[np.array(s_["head"]) == hh]):.0f}%'
                             for hh in range(H)))


if __name__ == '__main__':
    main(sys.argv[1])
