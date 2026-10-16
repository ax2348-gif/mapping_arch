"""Read the learned delta-rule write strength (beta) and decay (a) of a trained
SE(2) hybrid4-family checkpoint, per token kind and odometry symbol.

    python tools/read_beta.py results/se2d/hybrid4/xs/seed0_100k_qt/final.pt

Only works when the first block is the Gated DeltaNet layer (true for every
rung of hybrid4 and its ablations): the forward is stopped at that layer's
input, so nothing past the embeddings runs and no GPU kernel is needed.
beta and a are computed from the layer input exactly as the layer does.
Reports, separately for the left-to-right and right-to-left heads, the
distribution of beta (and the fraction above 1, i.e. updates with a
negative eigenvalue) for: observation patch tokens, odometry tokens by
turn (none / right / around / left) and by body-frame move, and query
tokens.
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


def main(path, batches=8):
    model, args, spec, task = load_checkpoint(path, 'cpu')
    assert task == 'se2d', 'SE(2) checkpoints only'
    t = tasks.get(task)
    config = tasks.config_from_args(task, spec, args)
    _, test_bank = load_banks(args['bank'], config)
    data = eval_set(config, test_bank, 64, batches, 4242)
    gdn = model.blocks[0].mixer
    assert type(gdn).__name__ == 'GatedDeltaNet', type(gdn).__name__
    DECAY_TAU = sys.modules[type(gdn).__module__].DECAY_TAU
    is_beta1 = 'beta = torch.sigmoid' in open(sys.modules[type(gdn).__module__].__file__).read()
    cap = {}

    def hook(mod, inp):
        cap['x'] = inp[0].detach()
        raise Stop
    h = gdn.register_forward_pre_hook(hook)
    rows = []
    for b in data:
        obs, odo, pose, tgt, qt = t['task'].prepare(b, 'cpu')
        try:
            model(obs, odo, pose, query_time=qt)
        except Stop:
            pass
        x = cap['x']                                             # (B, L, w), sequence order
        bsz, L, _ = x.shape
        with torch.no_grad():
            z = gdn.beta(x).float()
            beta = (torch.sigmoid(z) if is_beta1 else 2 * torch.sigmoid(z))  # (B, L, H)
            a = torch.exp(F.logsigmoid(gdn.decay(x).float()) / DECAY_TAU)
        # token labels in layout order, then permuted as the model does
        t1, q = obs.shape[1], qt.shape[1]
        kind = np.array(([0] * 9 + [1]) * t1)[:-1].tolist() + [2] * (q * 9)
        sym = np.full((bsz, t1 * 10 - 1 + q * 9), -1)
        dir_pos = np.arange(t1 - 1) * 10 + 9
        sym[:, dir_pos] = odo.numpy()
        order = model.token_order(qt, t1, q).numpy()
        kind = np.take_along_axis(np.broadcast_to(np.array(kind), sym.shape), order, 1)
        sym = np.take_along_axis(sym, order, 1)
        rows.append((kind, sym, beta.numpy(), a.numpy()))
    h.remove()
    kind = np.concatenate([r[0] for r in rows]); sym = np.concatenate([r[1] for r in rows])
    beta = np.concatenate([r[2] for r in rows]); a = np.concatenate([r[3] for r in rows])
    H = beta.shape[-1]; half = H // 2
    turn_names = ['none', 'right', 'around', 'left']
    move_names = ['fwd', 'right', 'back', 'left']
    print(f'{path}\n  beta = {"sigmoid" if is_beta1 else "2*sigmoid"}; {H} heads ({half} per direction)')
    for d, heads in (('left-to-right', range(half)), ('right-to-left', range(half, H))):
        print(f'  {d} heads:')
        groups = [('observation patches', kind == 0), ('query tokens', kind == 2)]
        for tn in range(4):
            groups.append((f'odometry, turn {turn_names[tn]}', (kind == 1) & (sym // 4 == tn)))
        for mv in range(4):
            groups.append((f'odometry, move {move_names[mv]}', (kind == 1) & (sym % 4 == mv)))
        for name, m in groups:
            v = beta[m][:, list(heads)].ravel(); av = a[m][:, list(heads)].ravel()
            if v.size == 0:
                continue
            q10, q50, q90 = np.percentile(v, [10, 50, 90])
            print(f'    {name:24s} beta median {q50:5.2f} (10-90%: {q10:4.2f}-{q90:4.2f})  '
                  f'beta>1: {100 * (v > 1).mean():5.1f}%   decay a median {np.median(av):.4f}')


if __name__ == '__main__':
    main(sys.argv[1])
