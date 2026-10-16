"""hybrid4_cdiag: ablation of hybrid4 with a complex-diagonal transition.

A copy of ``mapping/models/hybrid4.py`` with one change: the Gated DeltaNet mixer is
replaced by ``ComplexDiagonal``, identical except for the state
transition. hybrid4's transition ``a (I - beta k k^T)`` can rotate the
state through products of reflections, which do NOT commute.
ComplexDiagonal rotates each key-channel pair by a learned input-dependent
angle: it CAN rotate, but all its transitions commute, and it has no
erase. The question it answers: is "can rotate" enough for SE(2), or does
the non-commuting delta rule matter? With the agent's position as a
complex number z, an SE(2) step is z <- e^(i theta) z + move, so a
commuting rotation plus additive writes suffices in principle (literature
review, docs/literature_review.md).

Kernel: pre-rotated q and k on flash-linear-attention's chunk_simple_gla;
checked against the loop ``cdiag_loop``. Parameters differ slightly from
hybrid4: no beta projection, plus a 16-angle-per-head projection.

Parent docstring follows.
hybrid4: hybrid3.5 with the scan on the fused kernel.

A copy of ``mapping/models/hybrid3_5.py`` with one change: ``GatedDeltaNet.scan`` calls
``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` (flash-linear-attention
0.5.2, Triton) instead of the sequential ``delta_scan`` loop. The layer's
inputs are unchanged: L2-normalised q (pre-scaled, so the kernel scale is
1) and k, log decay g, and beta in (0, 2) passed directly, which the
kernel applies as given. Checked against ``delta_scan``: relative error
2e-3 in fp32 and 7e-3 in bf16 on outputs, 2-4e-3 on every gradient, at
the model's sequence length; about 100x faster. With the same weights,
hybrid4 and hybrid3.5 agree to the same tolerance.

Parent docstring follows.
hybrid3.5: hybrid3 with Gated DeltaNet in place of gated linear
attention, scan written by hand in PyTorch.

A copy of ``mapping/models/hybrid3.py`` with one change: the linear-attention mixer is a
bidirectional Gated DeltaNet with negative eigenvalues (``GatedDeltaNet``).

Why: the parent's scan updates its state by a positive scalar decay plus
an additive write, so an input can only shrink the stored state, never
rotate it. Placing an earlier frame relative to the current one on SE(2)
needs exactly that rotation every time the agent turns. The delta rule's
transition ``a (I - beta k k^T)`` with ``beta`` up to 2 is a reflection at
the extreme, and two reflections compose into a rotation (Grazzi et al.
2024, negative eigenvalues for state tracking). One odometry token can
supply one reflection; the short convolution lets the following tokens
see it and supply the second.

The scan is a plain sequential loop (``delta_scan``): slow, but it is the
reference ``hybrid4`` is checked against. Everything else (SwiGLU, RoPE
on the step index, tied embeddings, per-step query placement, ladder,
muP rule, init) is the parent's.

Audit additions: the short convolution mixes neighbouring tokens in
sequence order (a sequence-index operation, like the scan itself); beta,
the decay and the convolution are learned from token content. Nothing is
computed from the odometry outside learned layers.

Parent docstring follows.
hybrid3: hybrid2 with SwiGLU in place of the 4x GELU MLP.

A copy of ``mapping/models/hybrid2.py`` with one change: every block's MLP, after both the
gated-linear-attention and the softmax-attention mixers, is SwiGLU with
hidden width 8/3 w rounded up to a multiple of 32, parameter-matched to
the parent's 4x GELU MLP. Mixers, RoPE on the step index, tied patch and
query embeddings, per-step query placement, ladder, muP rule and
initialisation are the parent's. The audit is the parent's: SwiGLU adds
no positional or geometric input.

Parent docstring follows.
hybrid2: the hybrid with time carried by RoPE instead of learned tables.

Fork of ``hybrid.py`` at the positional interface, the way ``vanilla2``
forks ``vanilla``; the gated-linear-attention / softmax-attention stack,
ladder, muP rule and initialisation are the parent's. What changes:

1. **RoPE on the step index in every softmax-attention layer.** Queries
   and keys are rotated by the token's step t (observation t, direction
   t, or a query asked at t), head dim 32 as 16 rotary pairs, base
   10000. Attention scores therefore depend on the step difference
   between two tokens from the first update; the parent had no
   positional information in attention beyond a learned 33-entry table
   that starts as noise.
2. **No per-step tables.** The learned time embedding and the token-type
   embedding are removed. Observation tokens are the patch embedding
   plus one of nine learned patch embeddings shared by every step;
   query tokens are the offset features plus one of nine learned query
   patch embeddings shared by every query; direction tokens are the
   direction embedding alone. Token kind is carried by those tables,
   step by RoPE and by sequence order.
3. **Queries sit at their step.** A query asked at step t has its nine
   tokens directly after observation t (``token_order``), as the parent
   now does; with every query at the last step this is the original
   layout.

The recurrence is unchanged and still reads time from sequence order
only.

Audit (the parent's items 1, 4, 5, 6, 7, 9 hold; items 2, 3, 6a, 8 are
replaced):
2. Step index: enters only through RoPE in attention (a rotation of q and
   k by the token's step, a sequence index) and through sequence order.
   Nothing is computed from the directions to obtain it.
3. Token kind: implied by which embedding table a token uses.
6a. Query time: the query's step sets its RoPE position and its place in
   the sequence; both sequence indices.
8. No other positional bias in attention.

Parent docstring follows.
Hybrid transformer: blocks alternate gated linear attention with softmax
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
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
from fla.ops.simple_gla import chunk_simple_gla

from mapping.models import register

ROPE_BASE = 10000.0

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
    'softmax attention (w/64 pairs), each followed by a SwiGLU MLP with '
    'hidden 8/3 w rounded up to a multiple of 32. '
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


def delta_scan(q, k, v, g, beta):
    """Gated delta rule, left to right, as a plain sequential recurrence.

    ``S_t = a_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`` and
    ``o_t = S_t^T q_t``, with ``a_t = exp(g_t)``. Written as
    ``S' = a_t S``, ``S_t = S' + beta_t k_t (v_t - S'^T k_t)^T``.
    q, k: (B, L, H, dk) with k unit-norm; v: (B, L, H, dv); g, beta:
    (B, L, H). Runs in float32. This is the reference the fused kernel in
    ``hybrid4`` is checked against.
    """
    dtype = v.dtype
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    b, l, h, dk = k.shape
    S = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(l):
        S = S * g[:, t].exp()[..., None, None]
        kS = torch.einsum('bhk,bhkv->bhv', k[:, t], S)
        S = S + torch.einsum('bhk,bhv->bhkv', k[:, t] * beta[:, t, :, None],
                             v[:, t] - kS)
        out.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))
    return torch.stack(out, 1).to(dtype)


class GatedDeltaNet(nn.Module):
    """Bidirectional Gated DeltaNet with negative eigenvalues: heads
    [0, H/2) scan left-to-right, [H/2, H) right-to-left.

    Per head: a depthwise short convolution (width 4, causal in the head's
    own scan direction) on q, k, v, then SiLU; q and k L2-normalised; a
    scalar data-dependent decay ``a``; a write strength ``beta = 2 *
    sigmoid(.)`` in (0, 2), so ``I - beta k k^T`` ranges up to a
    reflection and products of two can rotate the state; per-head RMSNorm
    and a SiLU output gate on the scan output, as in the parent's GLA.
    """

    def __init__(self, width, heads, mult, std, out_std, conv=4):
        super().__init__()
        assert heads % 2 == 0, 'need an even number of heads (two directions)'
        self.heads, self.d, self.conv = heads, HEAD_DIM, conv
        inner = heads * HEAD_DIM
        self.q = MuLinear(width, inner, mult, std)
        self.k = MuLinear(width, inner, mult, std)
        self.v = MuLinear(width, inner, mult, std)
        self.short = nn.Conv1d(3 * inner, 3 * inner, conv, groups=3 * inner,
                               bias=False)
        self.gate = MuLinear(width, inner, mult, std)
        self.decay = MuLinear(width, heads, mult, std)
        nn.init.constant_(self.decay.bias, 3.0)   # a ~ 0.997 at init
        self.beta = MuLinear(width, heads, mult, std)
        nn.init.constant_(self.beta.bias, 0.0)    # beta = 1 at init
        self.norm = nn.RMSNorm(HEAD_DIM)
        self.out = MuLinear(inner, width, mult, out_std)

    def scan(self, q, k, v, g, beta):
        o, _ = chunk_gated_delta_rule(q.contiguous(), k.contiguous(),
                                      v.contiguous(), g.contiguous(),
                                      beta.contiguous(), scale=1.0)
        return o

    def forward(self, x):
        b, l, _ = x.shape
        h, d, half = self.heads, self.d, self.heads // 2
        qkv = torch.stack([self.q(x), self.k(x), self.v(x)], 2)  # (b,l,3,h*d)
        qkv = qkv.view(b, l, 3, h, d)
        g = (F.logsigmoid(self.decay(x).float()) / DECAY_TAU)     # (b,l,h)
        beta = 2 * torch.sigmoid(self.beta(x).float())            # (0, 2)
        # put each half in its own scan order, then a causal short conv
        fwd = qkv[:, :, :, :half]
        bwd = qkv[:, :, :, half:].flip(1)
        seq = torch.cat([fwd, bwd], 3).reshape(b, l, 3 * h * d)
        seq = F.pad(seq.transpose(1, 2), (self.conv - 1, 0))
        seq = F.silu(self.short(seq)).transpose(1, 2).view(b, l, 3, h, d)
        q, k, v = seq.unbind(2)
        q = F.normalize(q.float(), dim=-1).to(v.dtype) * d ** -0.5
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        gh = torch.cat([g[:, :, :half], g[:, :, half:].flip(1)], 2)
        bh = torch.cat([beta[:, :, :half], beta[:, :, half:].flip(1)], 2)
        o = self.scan(q, k, v, gh, bh)                            # (b,l,h,d)
        o = torch.cat([o[:, :, :half], o[:, :, half:].flip(1)], 2)
        o = self.norm(o).to(x.dtype) * F.silu(self.gate(x)).view(b, l, h, d)
        return self.out(o.reshape(b, l, h * d))


def cdiag_loop(q, k, v, g, theta):
    """Reference: complex-diagonal recurrence as a plain loop.

    State S (dk x dv) per head; the key space is 16 complex pairs
    (channels 2j, 2j+1). Per token: S <- exp(g) R(theta) S + k v^T, with
    R(theta) block-diagonal 2x2 rotations by theta_j (one angle per pair);
    o = S^T q. q, k: (B, L, H, dk); v: (B, L, H, dv); g: (B, L, H);
    theta: (B, L, H, dk/2). Float32. Used only to check the parallel form."""
    dtype = v.dtype
    q, k, v, g, theta = (x.float() for x in (q, k, v, g, theta))
    b, l, h, dk = k.shape
    S = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(l):
        c, s = theta[:, t].cos(), theta[:, t].sin()            # (b,h,dk/2)
        S0, S1 = S[:, :, 0::2], S[:, :, 1::2]                  # (b,h,dk/2,dv)
        R0 = c[..., None] * S0 - s[..., None] * S1
        R1 = s[..., None] * S0 + c[..., None] * S1
        S = torch.stack([R0, R1], 3).flatten(2, 3) * g[:, t].exp()[..., None, None]
        S = S + torch.einsum('bhk,bhv->bhkv', k[:, t], v[:, t])
        out.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))
    return torch.stack(out, 1).to(dtype)


def rotate_pairs(x, ang):
    """Rotate channel pairs (2j, 2j+1) of x (..., dk) by ang (..., dk/2)."""
    c, s = ang.cos(), ang.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([c * x0 - s * x1, s * x0 + c * x1], -1).flatten(-2)


def cdiag_parallel(q, k, v, g, theta):
    """The same recurrence in parallel form. Its transitions commute, so
    the product from s+1 to t is exp(G_t - G_s) R(Phi_t - Phi_s) with Phi
    the running sum of theta, and q_t^T R(Phi_t - Phi_s) k_s equals
    (R(-Phi_t) q_t)^T (R(-Phi_s) k_s). That is decayed linear attention on
    pre-rotated q and k, run on the fused simple-GLA kernel. Phi is the
    layer's own scan of its learned angles, as G is of its log decays."""
    phi = torch.cumsum(theta.float(), 1)                        # (B,L,H,dk/2)
    qr = rotate_pairs(q.float(), -phi).to(v.dtype)
    kr = rotate_pairs(k.float(), -phi).to(v.dtype)
    o, _ = chunk_simple_gla(qr.contiguous(), kr.contiguous(), v.contiguous(),
                            g=g.float().contiguous(), scale=1.0)
    return o


class ComplexDiagonal(nn.Module):
    """Ablation mixer: hybrid4's Gated DeltaNet with the transition replaced
    by a complex-diagonal (block-rotation) one. Same projections, causal
    short convolution per scan direction, L2-normalised q and k, decay,
    per-head RMSNorm and SiLU output gate, two scan directions. Per token,
    each of the 16 key-channel pairs is rotated by a learned input-dependent
    angle theta = pi * tanh(.) (near 0 at init) and decayed, then k v^T is
    added: the state can rotate, but all transitions commute, and there is
    no erase (no beta)."""

    def __init__(self, width, heads, mult, std, out_std, conv=4):
        super().__init__()
        assert heads % 2 == 0, 'need an even number of heads (two directions)'
        self.heads, self.d, self.conv = heads, HEAD_DIM, conv
        inner = heads * HEAD_DIM
        self.q = MuLinear(width, inner, mult, std)
        self.k = MuLinear(width, inner, mult, std)
        self.v = MuLinear(width, inner, mult, std)
        self.short = nn.Conv1d(3 * inner, 3 * inner, conv, groups=3 * inner,
                               bias=False)
        self.gate = MuLinear(width, inner, mult, std)
        self.decay = MuLinear(width, heads, mult, std)
        nn.init.constant_(self.decay.bias, 3.0)   # a ~ 0.997 at init
        self.angle = MuLinear(width, heads * HEAD_DIM // 2, mult, std)
        nn.init.zeros_(self.angle.bias)           # theta ~ 0 at init
        self.norm = nn.RMSNorm(HEAD_DIM)
        self.out = MuLinear(inner, width, mult, out_std)

    def scan(self, q, k, v, g, theta):
        return cdiag_parallel(q, k, v, g, theta)

    def forward(self, x):
        b, l, _ = x.shape
        h, d, half = self.heads, self.d, self.heads // 2
        qkv = torch.stack([self.q(x), self.k(x), self.v(x)], 2)  # (b,l,3,h*d)
        qkv = qkv.view(b, l, 3, h, d)
        g = (F.logsigmoid(self.decay(x).float()) / DECAY_TAU)     # (b,l,h)
        theta = math.pi * torch.tanh(self.angle(x).float()).view(b, l, h, d // 2)
        fwd = qkv[:, :, :, :half]
        bwd = qkv[:, :, :, half:].flip(1)
        seq = torch.cat([fwd, bwd], 3).reshape(b, l, 3 * h * d)
        seq = F.pad(seq.transpose(1, 2), (self.conv - 1, 0))
        seq = F.silu(self.short(seq)).transpose(1, 2).view(b, l, 3, h, d)
        q, k, v = seq.unbind(2)
        q = F.normalize(q.float(), dim=-1).to(v.dtype) * d ** -0.5
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        gh = torch.cat([g[:, :, :half], g[:, :, half:].flip(1)], 2)
        th = torch.cat([theta[:, :, :half], theta[:, :, half:].flip(1)], 2)
        o = self.scan(q, k, v, gh, th)                            # (b,l,h,d)
        o = torch.cat([o[:, :, :half], o[:, :, half:].flip(1)], 2)
        o = self.norm(o).to(x.dtype) * F.silu(self.gate(x)).view(b, l, h, d)
        return self.out(o.reshape(b, l, h * d))


def rope_cos_sin(pos, dim, base=ROPE_BASE):
    """``(cos, sin)`` of shape ``(b, l, dim/2)`` for integer positions
    ``pos (b, l)``: the standard rotary table, one angle per pair."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device).float()
                          / dim))
    ang = pos.float()[..., None] * inv                         # (b,l,dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs of channels of ``x (b, l, h, d)`` by the angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, :, None, :].to(x.dtype), sin[:, :, None, :].to(x.dtype)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Attention(nn.Module):
    """Full (non-causal) multi-head softmax attention with RoPE over the
    token's step index."""

    def __init__(self, width, heads, mult, std, out_std):
        super().__init__()
        self.heads = heads
        self.qkv = MuLinear(width, 3 * width, mult, std)
        self.out = MuLinear(width, width, mult, out_std)

    def forward(self, x, cos, sin):
        b, l, w = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.heads, -1).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2),
                                           k.transpose(1, 2),
                                           v.transpose(1, 2))
        return self.out(o.transpose(1, 2).reshape(b, l, w))


def swiglu_hidden(width):
    """8/3 width rounded up to a multiple of 32: parameter-matched to the
    parent's 4x GELU MLP (3 * w * h ~ 8 w^2)."""
    return 32 * math.ceil(8 * width / 3 / 32)


class MLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``. The product of two linear
    maps of the input lets a block compute bilinear functions of its input
    directly, which the parent's GELU MLP can only approximate."""

    def __init__(self, width, mult, std, out_std):
        super().__init__()
        hidden = swiglu_hidden(width)
        self.gate_up = MuLinear(width, 2 * hidden, mult, std)
        self.down = MuLinear(hidden, width, mult, out_std)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, mixer, width, mult, std, out_std):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(width)
        self.mlp = MLP(width, mult, std, out_std)

    def forward(self, x, cos, sin):
        h = self.norm1(x)
        h = (self.mixer(h, cos, sin) if isinstance(self.mixer, Attention)
             else self.mixer(h))
        x = x + h
        return x + self.mlp(self.norm2(x))


class Hybrid4Cdiag(nn.Module):
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
        self.patch_pos = nn.Embedding(9, width)      # shared by every step
        self.query_pos = nn.Embedding(9, width)      # shared by every query
        self.dir_emb = nn.Embedding(spec.directions, width)
        self.n_freq = 8
        self.offset_embed = nn.Linear(2 + 4 * self.n_freq, width)
        self.blocks = nn.ModuleList()
        for i in range(mixers):
            if i % 2 == 0:
                mixer = ComplexDiagonal(width, heads, mult, std, out_std)
            else:
                mixer = Attention(width, heads, mult, std, out_std)
            self.blocks.append(Block(mixer, width, mult, std, out_std))
        self.norm = nn.LayerNorm(width)
        self.readout = MuLinear(width, 9, mult, std, zero=True)
        for emb in (self.patch_pos, self.query_pos, self.dir_emb):
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
        qry_key = (query_time[:, :, None] * 30 + 10).expand(-1, -1, 9)
        qry_key = qry_key.reshape(b, -1)                           # (b, q*9)
        key = torch.cat([ctx_key[None].expand(b, -1), qry_key], 1).double()
        # ties (the nine tokens of a query, several queries at one step)
        # keep their layout order: query index, then patch index
        key = key + torch.arange(key.shape[1], device=dev, dtype=key.dtype) * 1e-4
        return torch.argsort(key, 1)

    def forward(self, obs, directions, query_offset, query_time=None):
        b, t1 = obs.shape[:2]
        q = query_offset.shape[1]
        w = self.width
        dev = obs.device
        if query_time is None:
            query_time = torch.full((b, q), t1 - 1, device=dev,
                                    dtype=torch.long)
        # observation tokens: (b, 33, 9 patches, 9 pixels)
        patches = (obs.view(b, t1, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                   .reshape(b, t1, 9, 9))
        idx = torch.arange(9, device=dev)
        steps = torch.arange(t1, device=dev)
        obs_tok = self.patch_embed(patches) + self.patch_pos(idx)
        # direction tokens, one per step, padded with a dummy after step 32
        dir_tok = F.pad(self.dir_emb(directions), (0, 0, 0, 1))   # (b,33,w)
        seq = torch.cat([obs_tok, dir_tok[:, :, None]], 2)        # (b,33,10,w)
        seq = seq.reshape(b, t1 * 10, w)[:, :-1]                  # drop pad
        # query tokens: nine per query
        qry = (self.offset_embed(self.offset_features(query_offset))[:, :, None]
               + self.query_pos(idx))                             # (b,q,9,w)
        x = torch.cat([seq, qry.reshape(b, q * 9, w)], 1)         # (b,401,w)
        # every token's step, in layout order, then the per-step placement
        ctx_pos = torch.cat([steps[:, None].expand(t1, 10)], 1).reshape(-1)[:-1]
        pos = torch.cat([ctx_pos[None].expand(b, -1),
                         query_time[:, :, None].expand(-1, -1, 9).reshape(b, -1)],
                        1)                                        # (b,401)
        order = self.token_order(query_time, t1, q)               # (b,401)
        x = torch.gather(x, 1, order[..., None].expand(-1, -1, w))
        pos = torch.gather(pos, 1, order)
        cos, sin = rope_cos_sin(pos, HEAD_DIM)
        for block in self.blocks:
            x = block(x, cos, sin)
        inverse = torch.argsort(order, 1)                        # back to layout
        x = torch.gather(x, 1, inverse[..., None].expand(-1, -1, w))
        y = self.readout(self.norm(x[:, -q * 9:]))                # (b,q*9,9)
        return (y.view(b, q, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                .reshape(b, q, 9, 9))


def build(size, spec):
    return Hybrid4Cdiag(spec, **SIZES[size])


register('hybrid4_cdiag')(sys.modules[__name__])
