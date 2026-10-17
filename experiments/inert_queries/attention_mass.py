"""Where do query rows attend in a trained se2d hybrid4 (attention layer)?
Mass on the anchor observation's 9 patches (same step as the query), on
other history, and on the query's own / other queries' tokens; inert vs mixed.

    python experiments/inert_queries/attention_mass.py CKPT [CKPT ...]
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import mapping.tasks as tasks  # noqa: E402
from mapping.checkpoint import load_checkpoint  # noqa: E402
from mapping.loader import eval_set, load_banks  # noqa: E402
from mapping.models.query_mode import QueryLayout  # noqa: E402
from mapping.models.se2d import hybrid4 as H  # noqa: E402

config, spec = tasks.config_and_spec('se2d', 32, 8, 0, 'uniform')
_, bank = load_banks('data/mnist.npz', config, 'mnist')
batch = eval_set(config, bank, 64, 1, 0)[0]
obs, odo, pose, target, qt = tasks.get('se2d')['task'].prepare(batch, 'cuda')
order = H.Hybrid4.token_order(qt, 33, 8)
lay = QueryLayout(order, 329, 8, 9)
steps = torch.cat([torch.arange(33).repeat_interleave(10)[:-1].cuda().expand(64, -1),
                   qt.repeat_interleave(9, 1)], 1).gather(1, order)       # sorted
is_obs = torch.cat([(torch.arange(330) % 10 != 9)[:-1].cuda().expand(64, -1),
                    torch.zeros(64, 72, dtype=torch.bool, device='cuda')], 1).gather(1, order)

for ck in sys.argv[1:]:
    net, args, _, _ = load_checkpoint(ck, 'cuda')
    caught = {}

    def hook(mod, inp, out, caught=caught):
        x, cos, sin = inp[:3]
        b, l, w = x.shape
        q, k, v = mod.qkv(x).view(b, l, 3, mod.heads, -1).unbind(2)
        q, k = H.apply_rope(q, cos, sin), H.apply_rope(k, cos, sin)
        s = torch.einsum('blhd,bmhd->bhlm', q, k) / q.shape[-1] ** 0.5
        if len(inp) > 3 and inp[3] is not None:
            s = s.masked_fill(~inp[3].attn_mask(mod.causal), float('-inf'))
        caught['p'] = s.softmax(-1)                                  # (b,h,L,L)
    for blk in net.blocks:
        if isinstance(blk.mixer, H.Attention):
            blk.mixer.register_forward_hook(hook)
    with torch.no_grad():
        y = net(obs, odo, pose, query_time=qt)
    p = caught['p'].mean(1)                                          # (b,L,L) head-avg
    qrow = ~lay.hist
    same_q = lay.group[:, :, None] == lay.group[:, None, :]
    anchor = is_obs[:, None, :] & (steps[:, None, :] == steps[:, :, None])
    def mass(m):
        return (p * m).sum(-1)[qrow].mean().item()
    other_hist = lay.hist[:, None, :] & ~anchor
    other_q = ~lay.hist[:, None, :] & ~same_q
    print(f'{ck}\n  query rows: anchor obs {mass(anchor):.3f}  other history {mass(other_hist):.3f}  '
          f'own query {mass(same_q & ~lay.hist[:, None, :]):.3f}  other queries {mass(other_q):.3f}'
          f'  (mse on this batch {F.mse_loss(y.float(), target):.4f})')
