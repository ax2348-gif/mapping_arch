"""Evaluate P5 (as frozen) and P5' (post hoc) of HYPOTHESIS.md on a trained
SE(2) hybrid4-family checkpoint whose first block is the Gated DeltaNet.

    python tools/probe_p5.py results/se2d/hybrid4/xs/seed0_100k_qt/final.pt

Runs on CPU. Recomputes the first layer's q, k, v, decay and beta exactly
as the layer does (tools/transition_probe.layer_quantities), runs the
delta rule as a loop to obtain the state S (dk x dv) after every token in
each head's scan order, then:

P5 (frozen): ridge probe from the LEFT-TO-RIGHT state read at a query token
  (anchor step t) to the pose of step s = t - lag relative to step t,
  lag 1..16. Targets: body-frame (dy, dx) of window s seen from pose t, and
  (cos, sin) of the heading change. Pass: R^2 >= 0.8 for every lag.
P5'(b): ridge probe from the RIGHT-TO-LEFT state read at observation s
  (after its last patch in that head's scan order) to the pose of step s
  relative to step t = s + lag, lag 1..16. Same targets and threshold.
P5'(a): for step pairs s < t with 1 <= t - s <= 16, the product of the
  right-to-left head's per-step transitions from t down to s has an
  eigenvalue (|lambda| > 0.5) within 10 degrees of the accumulated heading
  change |h_t - h_s| (0, 90 or 180 degrees; for 0, every significant
  eigenvalue must be within 10 degrees of 0). Pass: >= 95% of pairs.
Ridge: features are the flattened state (all heads of that direction),
standardised; lambda chosen on a validation split; R^2 reported on
held-out episodes (episode-level split).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402
from mapping import se2d  # noqa: E402
from transition_probe import Stop, layer_quantities  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def full_quantities(gdn, x):
    """q, v as well as k, g, beta, in each head's scan order."""
    mod = sys.modules[type(gdn).__module__]
    b, l, _ = x.shape
    h, d, half = gdn.heads, gdn.d, gdn.heads // 2
    qkv = torch.stack([gdn.q(x), gdn.k(x), gdn.v(x)], 2).view(b, l, 3, h, d)
    fwd, bwd = qkv[:, :, :, :half], qkv[:, :, :, half:].flip(1)
    seq = torch.cat([fwd, bwd], 3).reshape(b, l, 3 * h * d)
    src = open(mod.__file__).read()
    if 'ABLATION: no short conv' in src:
        seq = F.silu(seq).view(b, l, 3, h, d)
    else:
        seq = F.pad(seq.transpose(1, 2), (gdn.conv - 1, 0))
        seq = F.silu(gdn.short(seq)).transpose(1, 2).view(b, l, 3, h, d)
    q = F.normalize(seq[:, :, 0].float(), dim=-1) * d ** -0.5
    v = seq[:, :, 2].float()
    k, g, beta = layer_quantities(gdn, x)
    return q.numpy(), k, v.numpy(), g, beta


def states(k, v, g, beta):
    """S after every token: (B, L, H, dk, dv), plain delta-rule loop."""
    B, L, H, dk = k.shape
    S = np.zeros((B, H, dk, v.shape[-1]))
    out = np.empty((B, L, H, dk, v.shape[-1]), dtype=np.float32)
    for t in range(L):
        S = S * np.exp(g[:, t])[..., None, None]
        kS = np.einsum('bhk,bhkv->bhv', k[:, t], S)
        S = S + np.einsum('bhk,bhv->bhkv', k[:, t] * beta[:, t][..., None], v[:, t] - kS)
        out[:, t] = S
    return out


def ridge_r2(X, Y, groups, rng):
    """Episode-split ridge regression; returns R^2 per target column."""
    eps = np.unique(groups); rng.shuffle(eps)
    n = len(eps); tr, va, te = eps[:int(.6 * n)], eps[int(.6 * n):int(.8 * n)], eps[int(.8 * n):]
    m = lambda e: np.isin(groups, e)  # noqa: E731
    mu, sd = X[m(tr)].mean(0), X[m(tr)].std(0) + 1e-6
    Z = (X - mu) / sd
    ym = Y[m(tr)].mean(0)
    best = None
    for lam in (1e-1, 1, 10, 100, 1000, 1e4):
        A = Z[m(tr)]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Y[m(tr)] - ym))
        err = ((Z[m(va)] @ W + ym - Y[m(va)]) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, W)
    W = best[1]
    pred = Z[m(te)] @ W + ym
    yt = Y[m(te)]
    return 1 - ((pred - yt) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)


def rel_pose(pos, head, s, t):
    """Pose of window s seen from pose t: body-frame (dy, dx), cos, sin of
    the heading change."""
    body = se2d.rotate(pos[s] - pos[t], head[t])
    dh = (head[s] - head[t]) % 4
    return [body[0], body[1], np.cos(dh * np.pi / 2), np.sin(dh * np.pi / 2)]


def main(path, batches=8, seed=0):
    model, args, spec, task = load_checkpoint(path, 'cpu')
    t_ = tasks.get(task)
    config = tasks.config_from_args(task, spec, args)
    _, test_bank = load_banks(args['bank'], config)
    data = eval_set(config, test_bank, 64, batches, 4242)
    gdn = model.blocks[0].mixer
    assert type(gdn).__name__ == 'GatedDeltaNet'
    H, half, d = gdn.heads, gdn.heads // 2, gdn.d
    cap = {}

    def hook(mod, inp):
        cap['x'] = inp[0].detach()
        raise Stop
    hk = gdn.register_forward_pre_hook(hook)
    p5 = {lag: ([], [], []) for lag in range(1, 17)}
    p5b = {lag: ([], [], []) for lag in range(1, 17)}
    consist = {lag: [] for lag in range(1, 17)}
    ep_base = 0
    for bt in data:
        obs, odo, pose, tgt, qt = t_['task'].prepare(bt, 'cpu')
        pos = bt['positions'].numpy(); head = bt['headings'].numpy()
        try:
            model(obs, odo, pose, query_time=qt)
        except Stop:
            pass
        with torch.no_grad():
            q, k, v, g, beta = full_quantities(gdn, cap['x'])
        Sx = states(k, v, g, beta)                       # (B, L, H, dk, dv)
        B, L = k.shape[:2]
        t1, nq = obs.shape[1], qt.shape[1]
        layout = []
        for s in range(t1):
            layout += [('o', s)] * 9 + ([('d', s)] if s < t1 - 1 else [])
        order = model.token_order(qt, t1, nq).numpy()
        for i in range(B):
            qlab = [('q', int(qt[i, j]), j) for j in range(nq) for _ in range(9)]
            seq = [(layout + qlab)[o] for o in order[i]]
            rev = seq[::-1]
            ep = ep_base + i
            # P5: l2r state at the last token of each query's 9 tokens
            seen = {}
            for p, lab in enumerate(seq):
                if lab[0] == 'q':
                    seen[lab[2]] = (p, lab[1])
            for j, (p, tq) in seen.items():
                feat = Sx[i, p, :half].reshape(-1)
                for lag in range(1, 17):
                    s = tq - lag
                    if s >= 0:
                        p5[lag][0].append(feat); p5[lag][1].append(rel_pose(pos[i], head[i], s, tq)); p5[lag][2].append(ep)
            # P5'(b): r2l state after the last patch of observation s (in r2l order)
            last = {}
            for p, lab in enumerate(rev):
                if lab[0] == 'o':
                    last[lab[1]] = p
            for s, p in last.items():
                feat = Sx[i, p, half:].reshape(-1)
                for lag in range(1, 17):
                    tt = s + lag
                    if tt < t1:
                        p5b[lag][0].append(feat); p5b[lag][1].append(rel_pose(pos[i], head[i], s, tt)); p5b[lag][2].append(ep)
            # P5'(a): r2l products of per-step transitions between obs t and obs s
            first = {}
            for p, lab in enumerate(rev):
                if lab[0] == 'o' and lab[1] not in first:
                    first[lab[1]] = p
            for hh in range(half, H):
                for s in range(t1 - 1):
                    P = np.eye(d)
                    for tt in range(s + 1, min(s + 17, t1)):
                        # tokens strictly after obs tt's patches, up to obs (tt-1)'s last patch, in r2l order
                        a, b_ = last[tt] + 1, last[tt - 1] + 1
                        for j in range(a, b_):
                            kj = k[i, j, hh][:, None]
                            A = np.exp(g[i, j, hh]) * (np.eye(d) - beta[i, j, hh] * (kj @ kj.T))
                            P = A @ P
                        # P maps the state at obs tt to the state at obs s
                        ev = np.linalg.eigvals(P)
                        sig = ev[np.abs(ev) > 0.5]
                        ang = np.degrees(np.abs(np.angle(sig)))
                        want = 90 * ((head[i, tt] - head[i, s]) % 4)
                        want = 360 - want if want > 180 else want
                        ok = (len(sig) > 0 and np.all(ang < 10)) if want == 0 else bool(np.any(np.abs(ang - want) < 10))
                        consist[tt - s].append(ok)
        ep_base += B
    hk.remove()
    rng = np.random.default_rng(seed)
    print(path)
    for name, store, head_name in (("P5 (frozen): left-to-right state at query token", p5, 'l2r'),
                                   ("P5'(b): right-to-left state at observation s", p5b, 'r2l')):
        print(f'  {name}; R^2 for dy, dx, cos dh, sin dh')
        worst = 1.0
        for lag in range(1, 17):
            X = np.array(store[lag][0]); Y = np.array(store[lag][1], dtype=float); gr = np.array(store[lag][2])
            r2 = ridge_r2(X, Y, gr, rng)
            worst = min(worst, float(np.nanmin(r2)))
            if lag in (1, 2, 4, 8, 12, 16):
                print(f'    lag {lag:2d}: n={len(Y):5d}  ' + '  '.join(f'{x:5.2f}' for x in r2))
        print(f'    minimum over lags and targets: {worst:.2f}  -> {"PASS" if worst >= 0.8 else "FAIL"} (threshold 0.8)')
    print("  P5'(a): right-to-left multi-step product has its eigenpair at the accumulated heading change")
    allok = []
    for lag in range(1, 17):
        allok += consist[lag]
        if lag in (1, 2, 4, 8, 12, 16):
            print(f'    gap {lag:2d}: {100 * np.mean(consist[lag]):5.1f}% of pairs')
    frac = np.mean(allok)
    print(f'    overall: {100 * frac:.1f}% -> {"PASS" if frac >= 0.95 else "FAIL"} (threshold 95%)')


if __name__ == '__main__':
    main(sys.argv[1])
