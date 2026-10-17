"""Bug checks for query_mode=inert (se2d hybrid4 xs), on real se2d batches.

1. Read path: the attention mask gives every query row all history keys;
   the history-only conv's gather/scatter is aligned (layout tokens in
   order); query outputs change when history observations change, and
   their gradient w.r.t. the observations is nonzero; attention mass a
   query puts on history.
3. Overfit: one fixed batch, N steps, inert vs mixed (same init).
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402
from mapping.models.query_mode import QueryLayout  # noqa: E402
from mapping.models.se2d import hybrid4 as H  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
config, spec = tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')
_, bank = load_banks('data/mnist.npz', config, 'mnist')
batch = eval_set(config, bank, 64, 1, 0)[0]
obs, odo, pose, target, qt = tasks.get('se2d')['task'].prepare(batch, 'cuda')
print('batch', obs.shape, pose.shape, 'query times', qt[0].tolist())

# ---- 1a. mask and alignment on this batch
order = H.Hybrid4.token_order(qt, 33, 8)
lay = QueryLayout(order, 329, 8, 9)
m = lay.attn_mask()[:, 0]                                   # (b, L, L)
qrows = ~lay.hist
hist_keys = (m & lay.hist[:, None, :]).sum(-1)               # per row
own = (m & ~lay.hist[:, None, :]).sum(-1)
print('query rows: history keys allowed min/max', hist_keys[qrows].min().item(),
      hist_keys[qrows].max().item(), '| own-query keys min/max', own[qrows].min().item(),
      own[qrows].max().item())
print('history rows: history keys', hist_keys[lay.hist].min().item(), 'query keys',
      own[lay.hist].max().item())
layout_index = order.gather(1, lay.hist_idx)                  # layout token at each history slot
print('history gather is layout tokens 0..328 in order:',
      torch.equal(layout_index, torch.arange(329, device='cuda').expand(64, -1)))


def build(mode):
    torch.manual_seed(0)
    return H.Hybrid4(spec, **H.SIZES['xs'], query_mode=mode).cuda()


def train(net, steps, lr=3e-4):
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=0.1)
    out = []
    for s in range(steps):
        with torch.autocast('cuda', torch.bfloat16):
            loss = F.mse_loss(net(obs, odo, pose, query_time=qt).float(), target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if s % 100 == 0 or s == steps - 1:
            out.append((s, loss.item()))
    return out


def read_path(net, tag):
    net.eval()
    with torch.no_grad():
        y = net(obs, odo, pose, query_time=qt)
        o2 = obs.clone()
        o2[:, :] = obs[torch.roll(torch.arange(64), 1)]          # other episodes' observations
        y2 = net(o2, odo, pose, query_time=qt)
        d2 = net(obs, (odo + 5) % 16, pose, query_time=qt)
    o = obs.clone().requires_grad_(True)
    net(o, odo, pose, query_time=qt).pow(2).sum().backward()
    g = o.grad.abs().sum((-1, -2))                              # (b, 33)
    print(f'{tag}: |dy| obs swapped {(y - y2).abs().mean():.4f}, odometry changed '
          f'{(y - d2).abs().mean():.4f}, |y| {y.abs().mean():.4f}; grad wrt obs '
          f'{g.sum():.3e} (anchor-step share {g.gather(1, qt).sum() / g.sum() / 8 * 33:.2f}x uniform)')
    net.train()


for mode in ('mixed', 'inert'):
    net = build(mode)
    hist = train(net, int(sys.argv[1]) if len(sys.argv) > 1 else 500)
    print(mode, 'overfit loss', ' '.join(f'{s}:{l:.4f}' for s, l in hist))
    read_path(net, mode)
