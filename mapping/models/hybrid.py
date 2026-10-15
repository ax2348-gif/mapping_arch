"""Hybrid transformer: blocks alternate gated linear attention with softmax
attention, in the style of recent hybrid SSM/attention stacks (Jamba,
Samba, Zamba, Griffin, Based).

Hypothesis
----------
The sequence contains a stream of 32 categorical step tokens whose running
sum is the quantity every later gather has to be keyed on. A running sum is
a prefix sum, which softmax attention cannot express cheaply (it needs an
attention pattern of O(T) weight per token and a readout that does the
counting), whereas a gated linear recurrence with decay close to one
computes it in one layer with constant parameters: ``h_t = a_t h_{t-1} +
k_t v_t^T`` with ``a_t -> 1`` is exactly a cumulative sum of the ``k v``
outer products. Given such a layer, the softmax layers only have to do
content matching and copying, which is what they are good at. The model is
a vanilla pre-norm transformer with every other token mixer replaced by a
gated-linear-attention layer; nothing else changes.

Sequence layout (ViT-style tokenisation, 401 tokens):

    [obs_0: 9 patch tokens] [dir_0] [obs_1: 9] [dir_1] ... [obs_32: 9]
    [query_0: 9 patch tokens] ... [query_7: 9 patch tokens]

* Observation tokens: 3x3 patches of the 9x9 observation (nine per
  observation) through a linear patch embedding, plus a learned
  patch-index embedding (0..8) and a learned time-step embedding (0..32).
* Direction tokens: a learned embedding of the categorical direction, plus
  the time-step embedding of the step it starts from. They are placed in
  sequence order between the two observations they connect, so a
  left-to-right scan passes each direction token exactly once between the
  content it separates, and a right-to-left scan passes the complementary
  set. Interleaving is what makes the recurrence's prefix sum line up with
  the content tokens; a separate direction segment would force the scan to
  hold 32 partial sums instead of one.
* Query tokens: nine per query (one per output 3x3 patch, like the mask
  tokens of a masked-autoencoder decoder), each carrying a learned
  patch-index embedding plus an embedding of the raw query offset (Fourier
  features of the two scaled scalars through a linear layer). They sit at
  the end so the left-to-right scan state at a query token summarises the
  whole sequence. Each query token is decoded to its 3x3 patch.

Recurrence: gated linear attention with a scalar, data-dependent decay per
head (Mamba-2 / "gated retention" form), implemented as a chunked parallel
scan in float32 in pure PyTorch (chunk 32: intra-chunk masked matmul,
inter-chunk state carried sequentially over 13 chunks). It is
bidirectional in the Vision-Mamba sense: half the heads scan left-to-right
and half right-to-left, with no shared state between them; the offline
setting allows this and it means a left-to-right head integrates the step
stream from the start of the sequence while a right-to-left head integrates
it from the end, so the query tokens at the end of the sequence and the
content tokens before them both have a one-layer path to the summary they
need. The output of the scan is per-head RMS-normalised and multiplied by a
SiLU output gate (GLA), then projected.

Scaling: one-parameter family in the width ``w``: heads ``w/32`` (head dim
32, so half the heads per direction), key/value state ``32 x 32`` per head
(total recurrent state ``32 w``), mixer layers ``w/32`` (``w/64`` GLA+attn
pairs), MLP ratio 4. Learning rate: muP-style transfer implemented as
weight multipliers so the harness's single ``--lr`` is the base learning
rate at base width 256; hidden and output matrices carry a multiplier
``256 / w`` (their effective Adam step shrinks as ``1/w``), with
initialisation chosen so the effective weights start at the same scale on
every rung. Head dimension is fixed, so attention logit scaling needs no
change. Residual-branch output projections are additionally initialised
with ``1 / sqrt(2 * mixers)`` (GPT-2 depth rule); the readout is
zero-initialised.

Audit: where positional or geometric information enters
--------------------------------------------------------
1. Patch-index embedding (nn.Embedding, 9 entries): the index of a 3x3
   patch within an observation or an output. A sequence index, allowed.
2. Time-step embedding (nn.Embedding, 33 entries): the observation index
   t in 0..32, attached to the observation's tokens and to direction token
   t. A sequence index, allowed; it carries no information about moves.
3. Token-type embedding (3 entries: observation, direction, query).
4. Direction tokens: ``nn.Embedding(4, w)`` over the raw direction class.
   No one-hot arithmetic, no cumulative sum, no ``MOVES`` lookup. The only
   place directions are combined with each other is inside the learned
   recurrence (and attention), whose gates and projections are learned.
5. Sequence order: direction tokens are interleaved in time order between
   observation groups. Order is the sequence index; nothing is computed
   from the direction values to place tokens.
6a. Query time (optional input ``query_time``): the step a query is asked
   at, added to its tokens through the same time-step embedding as the
   observations (item 2), and used to place the query's tokens in the
   sequence directly after that step's observation (``token_order``).
   Both are sequence indices, allowed; nothing is computed from the
   directions. Absent (the original last-step tasks) nothing changes, so
   earlier checkpoints are unchanged; with every query at the last step
   the placement is the original layout.
6. Query offset: Fourier features (sin/cos at 8 octave-spaced frequencies
   per axis, plus the two scalars scaled by ``offset_max``) of the raw
   offset through one linear layer, added to the query tokens. The offset
   is never combined with directions or anything else by arithmetic; it is
   embedded as a raw input, which the contract explicitly allows.
7. Un-patchify: output patch tokens are placed in the 9x9 output by their
   patch index (a sequence index), the inverse of the input patching.
8. No positional bias in attention (no RoPE, no relative bias); the
   recurrence orders tokens by sequence index only.
9. The model never receives ``positions`` and nothing is indexed, scattered
   or gathered by a coordinate.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.models import register

HEAD_DIM = 32         # per-head key/value dim; heads = width / HEAD_DIM
BASE_WIDTH = 256      # muP base width: the harness lr is the lr at this width
CHUNK = 32            # chunk length of the parallel scan
DECAY_TAU = 16.0      # GLA: log a = logsigmoid(z) / tau keeps a close to 1

SIZES = {
    'xs': dict(width=64),
    's': dict(width=128),
    'm': dict(width=192),
    'l': dict(width=256),
    'xl': dict(width=384),
    'xxl': dict(width=512),
}

# The harness reads LR[size] when --lr is not given. It is deliberately the
# same on every rung: per-width learning-rate scaling is done inside the
# model by the muP weight multipliers (see MuLinear), which only touch the
# hidden and readout matrices, not embeddings, biases or norm gains.
LR = {size: 3e-4 for size in SIZES}

SCALING_RULE = (
    'One-parameter family in the width w (64, 128, 192, 256, 384, 512): '
    'heads = w/32 with head dim 32 fixed (half the heads scan left-to-right, '
    'half right-to-left); recurrent state per head 32x32 so total state is '
    '32w; token-mixer layers = w/32 alternating gated linear attention and '
    'softmax attention (w/64 pairs), each followed by a 4x GELU MLP. '
    'Parameters grow ~w^3 (0.39 w^3 + embeddings). Learning rate: LR[size] '
    'is 3e-4 on every rung and is the muP base lr at base width 256; hidden and '
    'readout matrices carry a weight multiplier 256/w, which under Adam '
    'scales their effective learning rate by 256/w, with init chosen so the '
    'effective weights start at std 0.5/sqrt(fan_in) on every rung; '
    'embeddings, biases and norm gains keep the base lr. Residual output '
    'projections are scaled by 1/sqrt(2 * mixers) at init; the readout is '
    'zero-initialised.')


class MuLinear(nn.Module):
    """Linear layer whose effective weight is ``mult * weight``. Under Adam
    the effective learning rate of the weight scales with ``mult``; the
    init is chosen so the effective weight starts with std ``std``."""

    def __init__(self, fan_in, fan_out, mult, std, bias=True, zero=False):
        super().__init__()
        self.mult = mult
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in))
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None
        if zero:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=std / mult)

    def forward(self, x):
        return F.linear(x, self.weight * self.mult, self.bias)


def chunked_scan(q, k, v, g, chunk=CHUNK):
    """Gated linear attention, left to right, as a chunked parallel scan.

    ``h_t = a_t h_{t-1} + k_t v_t^T``, ``o_t = q_t h_t`` with scalar
    per-head decay ``a_t = exp(g_t)``, ``g_t <= 0``.
    q, k: (B, L, H, dk)  v: (B, L, H, dv)  g: (B, L, H)  -> o: (B, L, H, dv)
    Runs in float32 regardless of autocast.
    """
    with torch.autocast('cuda', enabled=False):
        q, k, v, g = q.float(), k.float(), v.float(), g.float()
        b, l, h, dk = q.shape
        dv = v.shape[-1]
        pad = (-l) % chunk
        if pad:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            g = F.pad(g, (0, 0, 0, pad))          # log a = 0 -> a = 1
        n = (l + pad) // chunk
        q = q.view(b, n, chunk, h, dk)
        k = k.view(b, n, chunk, h, dk)
        v = v.view(b, n, chunk, h, dv)
        g = g.view(b, n, chunk, h)
        G = g.cumsum(2)                                      # (b,n,c,h)
        # intra-chunk: o_t += sum_{s<=t} exp(G_t - G_s) (q_t.k_s) v_s
        diff = G[:, :, :, None, :] - G[:, :, None, :, :]     # (b,n,t,s,h)
        mask = torch.ones(chunk, chunk, dtype=torch.bool,
                          device=q.device).tril()
        decay = torch.exp(diff.masked_fill(~mask[None, None, :, :, None],
                                           float('-inf')))
        scores = torch.einsum('bnthd,bnshd->bntsh', q, k) * decay
        out = torch.einsum('bntsh,bnshv->bnthv', scores, v)
        # chunk states: S_n = exp(G_last) S_{n-1} + sum_s exp(G_last-G_s) k v
        to_end = torch.exp(G[:, :, -1:, :] - G)               # (b,n,c,h)
        kv = torch.einsum('bnsh,bnshk,bnshv->bnhkv', to_end, k, v)
        chunk_decay = torch.exp(G[:, :, -1, :])              # (b,n,h)
        qg = q * torch.exp(G)[..., None]                      # (b,n,c,h,dk)
        state = q.new_zeros(b, h, dk, dv)
        inter = []
        for i in range(n):
            inter.append(torch.einsum('bthk,bhkv->bthv', qg[:, i], state))
            state = chunk_decay[:, i][:, :, None, None] * state + kv[:, i]
        out = out + torch.stack(inter, 1)
        return out.reshape(b, n * chunk, h, dv)[:, :l]


class GatedLinearAttention(nn.Module):
    """Bidirectional GLA: heads [0, H/2) scan left-to-right, [H/2, H)
    right-to-left. Scalar data-dependent decay per head, per-head RMSNorm
    and a SiLU output gate on the scan output."""

    def __init__(self, width, heads, mult, std, out_std):
        super().__init__()
        assert heads % 2 == 0, 'need an even number of heads (two directions)'
        self.heads, self.d = heads, HEAD_DIM
        inner = heads * HEAD_DIM
        self.q = MuLinear(width, inner, mult, std)
        self.k = MuLinear(width, inner, mult, std)
        self.v = MuLinear(width, inner, mult, std)
        self.gate = MuLinear(width, inner, mult, std)
        self.decay = MuLinear(width, heads, mult, std)
        nn.init.constant_(self.decay.bias, 3.0)   # a ~ 0.997 at init
        self.norm = nn.RMSNorm(HEAD_DIM)
        self.out = MuLinear(inner, width, mult, out_std)

    def forward(self, x):
        b, l, _ = x.shape
        h, d = self.heads, self.d
        q = self.q(x).view(b, l, h, d) * d ** -0.5
        k = self.k(x).view(b, l, h, d)
        v = self.v(x).view(b, l, h, d)
        g = F.logsigmoid(self.decay(x).float()) / DECAY_TAU   # (b,l,h) <= 0
        half = h // 2
        fwd = chunked_scan(q[:, :, :half], k[:, :, :half], v[:, :, :half],
                           g[:, :, :half])
        bwd = chunked_scan(q[:, :, half:].flip(1), k[:, :, half:].flip(1),
                           v[:, :, half:].flip(1), g[:, :, half:].flip(1))
        o = torch.cat([fwd, bwd.flip(1)], 2)                  # (b,l,h,d)
        o = self.norm(o).to(x.dtype) * F.silu(self.gate(x)).view(b, l, h, d)
        return self.out(o.reshape(b, l, h * d))


class Attention(nn.Module):
    """Full (non-causal) multi-head softmax attention, no positional bias."""

    def __init__(self, width, heads, mult, std, out_std):
        super().__init__()
        self.heads = heads
        self.qkv = MuLinear(width, 3 * width, mult, std)
        self.out = MuLinear(width, width, mult, out_std)

    def forward(self, x):
        b, l, w = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.heads, -1).unbind(2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2),
                                           k.transpose(1, 2),
                                           v.transpose(1, 2))
        return self.out(o.transpose(1, 2).reshape(b, l, w))


class MLP(nn.Module):
    def __init__(self, width, mult, std, out_std):
        super().__init__()
        self.fc1 = MuLinear(width, 4 * width, mult, std)
        self.fc2 = MuLinear(4 * width, width, mult, out_std)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, mixer, width, mult, std, out_std):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(width)
        self.mlp = MLP(width, mult, std, out_std)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Hybrid(nn.Module):
    def __init__(self, spec, width):
        super().__init__()
        assert spec.view == 9, 'tokeniser assumes 9x9 views (3x3 patches)'
        self.spec = spec
        self.width = width
        heads = width // HEAD_DIM
        mixers = width // HEAD_DIM
        mult = BASE_WIDTH / width
        std = 0.5 / math.sqrt(width)          # effective init of hidden W
        out_std = std / math.sqrt(2 * mixers)  # residual-branch outputs
        self.patch_embed = nn.Linear(9, width)
        self.patch_pos = nn.Embedding(9, width)
        self.time_pos = nn.Embedding(spec.steps + 1, width)
        self.type_emb = nn.Embedding(3, width)
        self.dir_emb = nn.Embedding(spec.directions, width)
        self.n_freq = 8
        self.offset_embed = nn.Linear(2 + 4 * self.n_freq, width)
        self.blocks = nn.ModuleList()
        for i in range(mixers):
            if i % 2 == 0:
                mixer = GatedLinearAttention(width, heads, mult, std, out_std)
            else:
                mixer = Attention(width, heads, mult, std, out_std)
            self.blocks.append(Block(mixer, width, mult, std, out_std))
        self.norm = nn.LayerNorm(width)
        self.readout = MuLinear(width, 9, mult, std, zero=True)
        for emb in (self.patch_pos, self.time_pos, self.type_emb,
                    self.dir_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def offset_features(self, query_offset):
        o = query_offset.float() / self.spec.offset_max           # (b,q,2)
        freqs = (math.pi * 2.0 ** torch.arange(self.n_freq,
                                               device=o.device).float())
        ang = o[..., None] * freqs                                # (b,q,2,F)
        return torch.cat([o, ang.sin().flatten(-2), ang.cos().flatten(-2)],
                         -1)

    @staticmethod
    def token_order(query_time, t1, q):
        """Permutation ``(b, t1*10 - 1 + q*9)`` placing query q's tokens
        after observation ``query_time[q]``: a sort of the layout
        [obs_0 x9, dir_0, ..., obs_T x9, query tokens] by (step, kind,
        index) with kinds observation < query < direction."""
        dev = query_time.device
        b = query_time.shape[0]
        steps = torch.arange(t1, device=dev)
        obs_key = (steps[:, None] * 30 + torch.arange(9, device=dev)).reshape(-1)
        dir_key = steps[:-1] * 30 + 20
        ctx_key = torch.cat([obs_key.view(t1, 9), F.pad(dir_key, (0, 1))[:, None]],
                            1).reshape(-1)[:-1]                     # (t1*10-1,)
        qry_key = (query_time[:, :, None] * 30 + 10
                   + torch.arange(9, device=dev)).reshape(b, -1)   # (b, q*9)
        key = torch.cat([ctx_key[None].expand(b, -1), qry_key], 1).float()
        # stable: break ties by layout position so queries keep their order
        key = key + torch.arange(key.shape[1], device=dev) * 1e-4
        return torch.argsort(key, 1)

    def forward(self, obs, directions, query_offset, query_time=None):
        b, t1 = obs.shape[:2]
        q = query_offset.shape[1]
        w = self.width
        dev = obs.device
        # observation tokens: (b, 33, 9 patches, 9 pixels)
        patches = (obs.view(b, t1, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                   .reshape(b, t1, 9, 9))
        idx = torch.arange(9, device=dev)
        steps = torch.arange(t1, device=dev)
        types = self.type_emb.weight
        obs_tok = (self.patch_embed(patches) + self.patch_pos(idx)
                   + self.time_pos(steps)[None, :, None] + types[0])
        # direction tokens, one per step, padded with a dummy after step 32
        dir_tok = (self.dir_emb(directions) + self.time_pos(steps[:-1])
                   + types[1])                                    # (b,32,w)
        dir_tok = F.pad(dir_tok, (0, 0, 0, 1))                    # (b,33,w)
        seq = torch.cat([obs_tok, dir_tok[:, :, None]], 2)        # (b,33,10,w)
        seq = seq.reshape(b, t1 * 10, w)[:, :-1]                  # drop pad
        # query tokens: nine per query
        qry = (self.offset_embed(self.offset_features(query_offset))[:, :, None]
               + self.patch_pos(idx) + types[2])                  # (b,q,9,w)
        x = torch.cat([seq, qry.reshape(b, q * 9, w)], 1)         # (b,401,w)
        if query_time is not None:                                # per-step queries
            # Each query's nine tokens sit in the sequence right after the
            # observation of its own step t (before direction t), so the
            # scan state they read is the prefix up to their anchor. With
            # every query at the last step this is the layout above.
            x = x + torch.cat([torch.zeros_like(seq),
                               self.time_pos(query_time)[:, :, None]
                               .expand(-1, -1, 9, -1).reshape(b, q * 9, w)],
                              1)
            order = self.token_order(query_time, t1, q)           # (b,401)
            x = torch.gather(x, 1, order[..., None].expand(-1, -1, w))
            for block in self.blocks:
                x = block(x)
            inverse = torch.argsort(order, 1)                    # back to layout
            x = torch.gather(x, 1, inverse[..., None].expand(-1, -1, w))
        else:
            for block in self.blocks:
                x = block(x)
        y = self.readout(self.norm(x[:, -q * 9:]))                # (b,q*9,9)
        return (y.view(b, q, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                .reshape(b, q, 9, 9))


def build(size, spec):
    return Hybrid(spec, **SIZES[size])


register('hybrid')(sys.modules[__name__])
