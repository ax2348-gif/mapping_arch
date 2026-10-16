"""hybrid4_time: hybrid4 with the recurrence restricted to the time axis.

In hybrid4 every Gated DeltaNet layer scans ALL tokens: each view's 9
patches in raster order, the odometry tokens and the query patches. The
patch order within a step is arbitrary, and delta-rule updates do not
commute, so the state depends on it. Here the recurrence sees only tokens
that live on the time axis; patches are handled as sets by attention
(RoPE on the step index, as before).

Layers (same count and alternation as hybrid4; xs: 2, s: 4):

    [time scan, attention, ..., time scan, LAST]

``LAST`` is the ablation switch (the user's): attention, exactly hybrid4's
(``hybrid4_time``), or hybrid4's full bidirectional Gated DeltaNet over ALL
tokens in hybrid4's token order (``hybrid4_time_full``, its own module, ``ARCH = {'last': 'gdn'}``).
The switch swaps one layer's operator; the layer count is unchanged. At
xs (2 layers) this means ``hybrid4_time`` = [time scan, attention] and
``hybrid4_time_full`` = [time scan, full GDN].

The time scan (``TimeScan``), per block:

1. **Time tokens.** For each step t, in hybrid4's placement order: a reader
   R_t, then one token per query asked at t, then the odometry token O_t
   (none after the last step). R_t is the mean of observation t's patch
   tokens; a query token is the mean of that query's patch tokens; O_t is
   the odometry token (all from the block's normalised input). xs/se2:
   33 + 8 + 32 = 73 tokens per episode against hybrid4's 401.
2. **Only odometry tokens write.** Readers and query tokens are read-only:
   beta = 0 and no decay (g = 0), so they cannot change the state. The
   state holds composed motion only (a pose integrator); patch content
   never enters it.
3. **One rotation per odometry token.** FLA's gated DeltaProduct with 2
   Householder steps per token: O_t gives two keys, two values and two
   betas in (0, 2), so its transition (I - b2 k2 k2^T)(I - b1 k1 k1^T)
   can be a rotation (two reflections, det +1). hybrid4 had to get its
   second reflection from the patches after the odometry token through the
   short convolution; there is no convolution here.
4. **Both directions**, as hybrid4: heads [0, H/2) scan left to right,
   [H/2, H) right to left (the time sequence reversed).
5. **Injection.** The scan output (per-head RMSNorm, SiLU gate and output
   projection, as hybrid4's layer) at R_t is added to every patch token of
   observation t, at a query token to every token of that query, and at
   O_t to the odometry token. That is the block's residual update; then
   hybrid4's SwiGLU MLP.

The main sequence stays in layout order; attention is
permutation-equivariant apart from RoPE on the step index, so token order
matters only to the full GDN layer, which gathers into hybrid4's order.

Audit: hybrid4's items; patches enter the recurrence only through the
pooled, read-only reader and query tokens, never as a sequence.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.models.se2d import register
from mapping.models.se2d.hybrid4 import (BASE_WIDTH, DECAY_TAU, HEAD_DIM,  # noqa: F401
                                        LR, SCALING_RULE, SIZES, Attention,
                                        Block, GatedDeltaNet, Hybrid4,
                                        MuLinear, rope_cos_sin)

HOUSEHOLDER = 2          # reflections per odometry token: one rotation


def delta_product(q, k, v, g, beta, n=HOUSEHOLDER):
    """Gated DeltaProduct, ``k, v, beta`` interleaved (n per token). The FLA
    kernel on CUDA in bf16/fp16; FLA's float32 reference loop otherwise
    (tests, CPU)."""
    if q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
        from fla.ops.gated_delta_product import chunk_gated_delta_product
        o, _ = chunk_gated_delta_product(
            q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(),
            beta.contiguous().to(q.dtype), num_householder=n, scale=1.0)
        return o
    from fla.ops.gated_delta_product.naive import \
        naive_recurrent_gated_delta_product
    o, _ = naive_recurrent_gated_delta_product(
        q.float(), k.float(), v.float(), g.float(), beta.float(), 1.0, None,
        num_householder=n)
    return o.to(q.dtype)


def time_layout(query_time, t1):
    """Scan order of the time tokens. Returns ``(order, step, kind)``, each
    ``(b, L)``, L = t1 + q + t1 - 1: ``order`` indexes the concatenation
    [readers (t1) | queries (q) | odometry (t1 - 1)]; kind 0 reader,
    1 query, 2 odometry; within a step reader < queries < odometry
    (hybrid4's placement: observation, queries, direction)."""
    b, q = query_time.shape
    dev = query_time.device
    steps = torch.arange(t1, device=dev)
    st = torch.cat([steps[None].expand(b, -1), query_time,
                    steps[None, :-1].expand(b, -1)], 1)
    kd = torch.cat([torch.zeros(b, t1, device=dev, dtype=torch.long),
                    torch.ones(b, q, device=dev, dtype=torch.long),
                    torch.full((b, t1 - 1), 2, device=dev, dtype=torch.long)], 1)
    n = st.shape[1]
    key = (st * 3 + kd) * n + torch.arange(n, device=dev)
    order = torch.argsort(key, 1)
    return order, torch.gather(st, 1, order), torch.gather(kd, 1, order)


def time_context(obs_idx, odo_idx, qry_idx, query_time, length):
    """Gather/scatter between a main sequence of ``length`` tokens and the
    time sequence. ``obs_idx (t1, P)``, ``odo_idx (t1-1,)``, ``qry_idx
    (q, P)`` are the main-sequence positions of each observation patch,
    odometry token and query token."""
    t1, P = obs_idx.shape
    q = qry_idx.shape[0]
    order, step, kind = time_layout(query_time, t1)
    inv = torch.argsort(order, 1)

    def gather(hn):
        b, _, w = hn.shape
        obs = hn[:, obs_idx.reshape(-1)].view(b, t1, P, w).mean(2)
        qry = hn[:, qry_idx.reshape(-1)].view(b, q, -1, w).mean(2)
        src = torch.cat([obs, qry, hn[:, odo_idx]], 1)
        return torch.gather(src, 1, order[..., None].expand(-1, -1, w))

    def scatter(y):
        b, _, w = y.shape
        src = torch.gather(y, 1, inv[..., None].expand(-1, -1, w))
        rd, qy, od = src[:, :t1], src[:, t1:t1 + q], src[:, t1 + q:]
        out = y.new_zeros(b, length, w)
        out[:, obs_idx.reshape(-1)] = rd[:, :, None].expand(-1, -1, P, -1).reshape(b, -1, w)
        out[:, qry_idx.reshape(-1)] = qy[:, :, None].expand(-1, -1, qry_idx.shape[1], -1).reshape(b, -1, w)
        out[:, odo_idx] = od
        return out

    return {'gather': gather, 'scatter': scatter, 'write': kind == 2,
            'tkind': kind, 'tstep': step, 'torder': order}


class TimeScan(nn.Module):
    """The time-axis recurrence of one block (see the module docstring)."""

    def __init__(self, width, heads, mult, std, out_std, n=HOUSEHOLDER):
        super().__init__()
        assert heads % 2 == 0, 'need an even number of heads (two directions)'
        self.heads, self.d, self.n = heads, HEAD_DIM, n
        inner = heads * HEAD_DIM
        self.q = MuLinear(width, inner, mult, std)
        self.k = MuLinear(width, n * inner, mult, std)
        self.v = MuLinear(width, n * inner, mult, std)
        self.gate = MuLinear(width, inner, mult, std)
        self.decay = MuLinear(width, heads, mult, std)
        nn.init.constant_(self.decay.bias, 3.0)   # a ~ 0.997 at init
        self.beta = MuLinear(width, n * heads, mult, std)
        nn.init.constant_(self.beta.bias, 0.0)    # beta = 1 at init
        self.norm = nn.RMSNorm(HEAD_DIM)
        self.out = MuLinear(inner, width, mult, out_std)

    def quantities(self, x, write):
        """Per time token, in scan order: q (b,l,h,d), k and v (b,l,n,h,d),
        g (b,l,h), beta (b,l,n,h); read-only tokens have g = beta = 0."""
        b, l, _ = x.shape
        h, d, n = self.heads, self.d, self.n
        q = self.q(x).view(b, l, h, d)
        k = self.k(x).view(b, l, n, h, d)
        v = self.v(x).view(b, l, n, h, d)
        g = F.logsigmoid(self.decay(x).float()) / DECAY_TAU * write[..., None]
        beta = (2 * torch.sigmoid(self.beta(x).float()).view(b, l, n, h)
                * write[..., None, None])
        q = F.normalize(q.float(), dim=-1).to(v.dtype) * d ** -0.5
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        return q, k, v, g, beta

    def forward(self, x, ctx):
        b, l, _ = x.shape
        h, d, n, half = self.heads, self.d, self.n, self.heads // 2
        q, k, v, g, beta = self.quantities(x, ctx['write'])
        # left-to-right heads as they are; right-to-left heads reversed in time
        qh = torch.cat([q[:, :, :half], q[:, :, half:].flip(1)], 2)
        kh = torch.cat([k[:, :, :, :half], k[:, :, :, half:].flip(1)], 3)
        vh = torch.cat([v[:, :, :, :half], v[:, :, :, half:].flip(1)], 3)
        gh = torch.cat([g[:, :, :half], g[:, :, half:].flip(1)], 2)
        bh = torch.cat([beta[:, :, :, :half], beta[:, :, :, half:].flip(1)], 3)
        o = delta_product(qh, kh.reshape(b, l * n, h, d),
                          vh.reshape(b, l * n, h, d), gh,
                          bh.reshape(b, l * n, h), n)                  # (b,l,h,d)
        o = torch.cat([o[:, :, :half], o[:, :, half:].flip(1)], 2)
        o = self.norm(o).to(x.dtype) * F.silu(self.gate(x)).view(b, l, h, d)
        return self.out(o.reshape(b, l, h * d))


class TimeBlock(nn.Module):
    """Pre-norm block: the time scan on the gathered time tokens, injected
    back into the main sequence; then hybrid4's SwiGLU MLP (``Block``'s)."""

    def __init__(self, width, heads, mult, std, out_std):
        super().__init__()
        self.mixer = TimeScan(width, heads, mult, std, out_std)
        ref = Block(self.mixer, width, mult, std, out_std)
        self.norm1, self.norm2, self.mlp = ref.norm1, ref.norm2, ref.mlp

    def forward(self, x, cos, sin, ctx):
        y = self.mixer(ctx['gather'](self.norm1(x)), ctx)
        x = x + ctx['scatter'](y)
        return x + self.mlp(self.norm2(x))


class AttnBlock(nn.Module):
    """hybrid4's attention block, unchanged."""

    def __init__(self, width, heads, mult, std, out_std):
        super().__init__()
        self.block = Block(Attention(width, heads, mult, std, out_std),
                           width, mult, std, out_std)

    def forward(self, x, cos, sin, ctx):
        return self.block(x, cos, sin)


class FullBlock(nn.Module):
    """hybrid4's Gated DeltaNet block over ALL tokens, in hybrid4's token
    order (``ctx['order']``), unchanged."""

    def __init__(self, width, heads, mult, std, out_std):
        super().__init__()
        self.block = Block(GatedDeltaNet(width, heads, mult, std, out_std),
                           width, mult, std, out_std)

    def forward(self, x, cos, sin, ctx):
        w = x.shape[-1]
        y = self.block(torch.gather(x, 1, ctx['order'][..., None].expand(-1, -1, w)),
                       None, None)
        return torch.gather(y, 1, ctx['inverse'][..., None].expand(-1, -1, w))


def layer_blocks(width, last):
    """hybrid4's layer count and alternation, with the delta-rule layers as
    time scans and the LAST layer ``'attention'`` or ``'gdn'``."""
    assert last in ('attention', 'gdn')
    heads = layers = width // HEAD_DIM
    mult = BASE_WIDTH / width
    std = 0.5 / math.sqrt(width)
    out_std = std / math.sqrt(2 * layers)
    blocks = nn.ModuleList()
    for i in range(layers):
        if i == layers - 1:
            cls = FullBlock if last == 'gdn' else AttnBlock
        else:
            cls = TimeBlock if i % 2 == 0 else AttnBlock
        blocks.append(cls(width, heads, mult, std, out_std))
    return blocks


class Hybrid4Time(Hybrid4):
    """hybrid4's tokeniser, embeddings and readout (inherited); new blocks;
    the main sequence stays in layout order."""

    def __init__(self, spec, width, last='attention'):
        super().__init__(spec, width)
        self.last = last
        self.blocks = layer_blocks(width, last)

    def forward(self, obs, odometry, query_pose, query_time=None):
        directions = odometry
        query_offset, turn = query_pose[..., :2], query_pose[..., 2]
        b, t1 = obs.shape[:2]
        q = query_offset.shape[1]
        w = self.width
        dev = obs.device
        if query_time is None:
            query_time = torch.full((b, q), t1 - 1, device=dev, dtype=torch.long)
        patches = (obs.view(b, t1, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                   .reshape(b, t1, 9, 9))
        idx = torch.arange(9, device=dev)
        steps = torch.arange(t1, device=dev)
        obs_tok = self.patch_embed(patches) + self.patch_pos(idx)
        dir_tok = F.pad(self.dir_emb(directions), (0, 0, 0, 1))
        seq = torch.cat([obs_tok, dir_tok[:, :, None]], 2).reshape(b, t1 * 10, w)[:, :-1]
        qry = (self.offset_embed(self.offset_features(query_offset))[:, :, None]
               + self.turn_emb(turn)[:, :, None] + self.query_pos(idx))
        x = torch.cat([seq, qry.reshape(b, q * 9, w)], 1)          # layout order
        n_ctx = t1 * 10 - 1
        ctx_pos = steps[:, None].expand(t1, 10).reshape(-1)[:-1]
        pos = torch.cat([ctx_pos[None].expand(b, -1),
                         query_time[:, :, None].expand(-1, -1, 9).reshape(b, -1)], 1)
        cos, sin = rope_cos_sin(pos, HEAD_DIM)                    # layout order
        obs_idx = (steps[:, None] * 10 + idx).to(dev)             # (t1, 9)
        odo_idx = steps[:-1] * 10 + 9                             # (T,)
        qry_idx = n_ctx + torch.arange(q * 9, device=dev).view(q, 9)
        ctx = time_context(obs_idx, odo_idx, qry_idx, query_time, x.shape[1])
        if self.last == 'gdn':
            order = self.token_order(query_time, t1, q)
            ctx['order'], ctx['inverse'] = order, torch.argsort(order, 1)
            # the full GDN layer needs RoPE-free hybrid4 order only
        for block in self.blocks:
            x = block(x, cos, sin, ctx)
        y = self.readout(self.norm(x[:, -q * 9:]))
        return (y.view(b, q, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                .reshape(b, q, 9, 9))


def build(size, spec):
    return Hybrid4Time(spec, **SIZES[size])


register('hybrid4_time')(sys.modules[__name__])
