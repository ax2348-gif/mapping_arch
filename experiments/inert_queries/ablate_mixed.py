"""Which query path does a trained MIXED se2d hybrid4 depend on? Apply each
inert component alone at eval (no retraining) and measure held-out mse by
distance: (write) query tokens read-only in the delta rule (beta = g = 0);
(conv) history conv over history only + no conv for queries; (attn) the
inert attention mask; (all) full inert.

    python experiments/inert_queries/ablate_mixed.py CKPT
"""
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402
from mapping.models.se2d import hybrid4 as H  # noqa: E402

FLAGS = dict(write=0, conv=0, attn=0)


def gdn_forward_inert(self, x, layout):
    b, l, _ = x.shape
    h, d = self.heads, self.d
    qkv = torch.stack([self.q(x), self.k(x), self.v(x)], 2).view(b, l, 3, h, d)
    write = layout.hist.float()[..., None] if FLAGS['write'] else 1.0
    g = F.logsigmoid(self.decay(x).float()) / H.DECAY_TAU * write
    beta = 2 * torch.sigmoid(self.beta(x).float()) * write
    if FLAGS['conv']:
        seq = layout.scatter(F.silu(qkv), self.short_conv(layout.gather(qkv)))
    else:
        seq = self.short_conv(qkv)
    q, k, v = self._to_scan_order(seq, 3).unbind(2)
    q = F.normalize(q.float(), dim=-1).to(v.dtype) * d ** -0.5
    k = F.normalize(k.float(), dim=-1).to(v.dtype)
    o = self.scan(q, k, v, self._to_scan_order(g, 2), self._to_scan_order(beta, 2))
    o = self._to_scan_order(o, 2)
    o = self.norm(o).to(x.dtype) * F.silu(self.gate(x)).view(b, l, h, d)
    return self.out(o.reshape(b, l, h * d))


attn_forward = H.Attention.forward


def attn(self, x, cos, sin, layout=None):
    return attn_forward(self, x, cos, sin, layout if FLAGS['attn'] else None)


H.GatedDeltaNet.forward_inert = gdn_forward_inert
H.Attention.forward = attn

net, args, spec, task = load_checkpoint(sys.argv[1], 'cuda')
config = tasks.config_from_args(task, spec, args)
_, bank = load_banks(args.get('bank') or 'data/mnist.npz', config, 'mnist')
batches = eval_set(config, bank, 64, 16, args['seed'])
prepare = tasks.get(task)['task'].prepare
for name, f in [('mixed (as trained)', None), ('write only', dict(write=1, conv=0, attn=0)),
                ('conv only', dict(write=0, conv=1, attn=0)), ('attn mask only', dict(write=0, conv=0, attn=1)),
                ('all three (inert)', dict(write=1, conv=1, attn=1))]:
    net.query_mode = 'mixed' if f is None else 'inert'
    FLAGS.update(f or {})
    errs, dists = [], []
    with torch.no_grad():
        for bt in batches:
            obs, odo, pose, target, qt = prepare(bt, 'cuda')
            errs.append((net(obs, odo, pose, query_time=qt).float() - target).pow(2).mean((-1, -2)).flatten())
            dists.append((pose[..., :2].float().norm(dim=-1) / spec.step_px).flatten())
    e, d = torch.cat(errs), torch.cat(dists)
    r2 = math.sqrt(2) + 1e-6
    print(f'{name:20s} all {e.mean():.4f}  near {e[d == 0].mean():.4f}  '
          f'mid {e[(d > 0) & (d <= r2)].mean():.4f}  far {e[d > r2].mean():.4f}')
