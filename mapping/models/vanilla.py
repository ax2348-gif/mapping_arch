"""The arm under interrogation: a 2026-default general-purpose transformer.

Nothing here knows the task is about space. The episode is turned into one
flat token sequence and a standard pre-norm transformer (RoPE over the
sequence index, multi-head attention with QK-norm, SwiGLU MLP, RMSNorm,
GPT-2-style init, no biases, no dropout) is run over it with full
bidirectional attention, which the offline setting allows. Queries are
extra tokens at the end of the same sequence and are read out from their
own final hidden state. This is the design a careful engineer reaches when
told "throw a standard transformer at this sequence", built to be strong
rather than minimal: the point of the study is whether *this* scales on
the task, so it must not be a strawman.

Token layout (337 tokens for the default spec, fixed length):

    [obs_0: 9 patch tokens][dir_0][obs_1: 9][dir_1] ... [dir_31][obs_32: 9]
    [query_0] ... [query_7]

Tokenisation choices and why:

* Observations are split into 3x3 patches, giving 9 tokens per 9x9 window
  (ViT-style). A patch token is a linear projection of its 9 pixels plus a
  learned patch-index embedding (which of the nine patches it is). This is
  the standard image tokenisation; it keeps the per-token dimensionality
  small (9 pixels) so the token's content is not dominated by a huge
  linear projection, lets attention address parts of a window, and is the
  same 3x3 grid the related marlenv work uses. A frame-level token (one
  81-pixel token per window) is the alternative; it is cheaper (41 tokens)
  but forces the whole window through one vector and leaves attention
  nothing finer than a window to point at. Patches are the natural ViT
  choice and are what the vision scaling literature does.
* A step direction is a learned embedding of its 0..3 index, placed in the
  sequence between the two windows it connects. It is a token like any
  other; nothing is computed from it outside the network.
* A query is a token whose content is an embedding of its raw (dy, dx)
  offset: sinusoidal Fourier features of the offset scaled to [-1, 1] at
  nine octaves (wavelengths from the full range down to about two pixels)
  followed by a two-layer MLP, the same recipe used for continuous scalar
  conditioning (diffusion timesteps, NeRF inputs). A learned table over the
  577 values per axis would also be allowed; the Fourier/MLP form gives
  the network ordinal structure between neighbouring offsets for free,
  which matters because a one-pixel change moves the target.
* A learned token-type embedding (patch / direction / query) is added to
  every token so the three kinds are distinguishable by content.

Readout: query tokens sit in the same sequence and the same attention
stack (the MAE-decoder / BERT-[MASK] pattern: a token with a learned
content embedding that attends to the context and is decoded from its own
final state). This is chosen over a separate cross-attention decoder
(Perceiver-IO style) because it is the more standard option, uses one
stack of weights, and lets the queries participate from layer one rather
than only after the encoder has finished. The head is final RMSNorm then
a linear map from width to the 81 output pixels.

Positions: RoPE over the flat sequence index (0..336) in every attention
layer, standard base 10000, plus a learned per-step embedding (33 entries,
indexed by the time step t of the observation and of the direction token
that leaves it) and the learned patch-index embedding (index of a patch
inside its observation). RoPE carries relative sequence position; the
per-step table gives each step an absolute key as well, the ViT/BERT
convention for a fixed-length sequence. Both are sequence-index only.

Scaling: head dimension fixed at 64; width = 64 * heads; depth = heads + 2.
So a rung is fully determined by its width, and parameters grow roughly as
width cubed (12 * width^2 per block times width / 64 + 2 blocks); the +2
is a depth floor so the low end has four layers rather than two. The
ladder runs from width 128 (0.9M) to width 768 (99M), a factor of about
110.
Learning rate: the standard parameterisation is used with the harness's
single AdamW learning rate (3e-4, warmup + cosine) at every rung; see
SCALING_RULE for why that is reasonable over this width range and for the
muP-style alternative.

Audit: where positional or geometric information enters, and why each is
within the rules.

1. ``patch_pos`` (learned, 9 entries): the index of a 3x3 patch within its
   own 9x9 observation. Keyed on patch index within an observation, which
   the contract names as allowed. It is the same for every observation and
   carries nothing about where the observation is in the world.
2. RoPE in ``Attention``: keyed on the flat sequence index of each token
   (obs t patch p is at 10 t + p, direction t at 10 t + 9, query q at
   329 + q). Pure time/sequence index, never anything computed from the
   directions or the offsets.
2b. ``step_embed`` (learned, T + 1 = 33 entries): the time step t, added
   to the nine patch tokens of observation t and to direction token t.
   Keyed on the sequence (time) index only, the learned-absolute analogue
   of RoPE above; it carries nothing about where step t is in the world.
3. ``dir_embed`` (learned, 4 entries): the direction index as a token
   content embedding. The directions are never summed, indexed with, or
   turned into coordinates outside learned layers; they are inputs the
   attention stack has to integrate itself.
4. ``offset_embed``: Fourier features of the raw query offset divided by
   ``spec.offset_max`` (a constant), then an MLP. The offset is used only
   as the content of the query token; it is not combined by arithmetic
   with the directions, the observations, or any position, and it never
   keys an attention bias or a gather.
5. ``type_embed`` (learned, 3 entries): the token's kind. Not positional.
6. Patch extraction: a fixed reshape of each window into its nine 3x3
   patches, applied identically to every observation; no coordinates are
   computed and nothing is placed by position.

Nothing reads ``positions``. No cumulative sums, no canvas, no scatter or
gather at computed coordinates, no attention bias keyed on anything but
the sequence index.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.models import register

HEAD_DIM = 64

SIZES = {
    'xs': dict(width=128),
    's': dict(width=192),
    'm': dict(width=256),
    'l': dict(width=384),
    'xl': dict(width=512),
    'xxl': dict(width=768),
}

SCALING_RULE = (
    'One-parameter family in width: head dim is fixed at 64, heads = '
    'width / 64, depth = heads + 2 (so 128/2/4, 192/3/5, 256/4/6, '
    '384/6/8, 512/8/10, 768/12/14 for width/heads/depth; the +2 is a depth '
    'floor so the smallest rung has four layers), SwiGLU hidden = 8/3 '
    'width rounded up to a multiple of 64. Parameters grow about as '
    'width^3 at the top of the ladder. '
    'Learning rate: standard parameterisation, residual-branch output '
    'projections initialised with std 0.02 / sqrt(2 depth), and the '
    'harness AdamW rate 3e-4 (warmup, cosine) at every rung. This is the '
    'value GPT-3 used at width 1024 and half what GPT-2/GPT-3 used at 768, '
    'so it is inside the usual stable range for the whole ladder rather '
    'than tuned per rung. A muP-style rule (lr proportional to 256 / '
    'width for hidden matrices) is the stated alternative; the harness '
    'has a single --lr, so it would be applied by passing --lr per rung.')


def _rope_cache(positions, dim, base=10000.0):
    """``(cos, sin)`` of shape ``(positions, dim / 2)`` for standard RoPE."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    ang = positions.float()[:, None] * inv[None]
    return ang.cos(), ang.sin()


def _apply_rope(x, cos, sin):
    """Rotate pairs of channels; ``x`` is ``(B, H, N, D)``, cos/sin ``(N, D/2)``."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        b, n, w = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # (B, H, N, D)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v)           # full attention
        return self.proj(y.transpose(1, 2).reshape(b, n, w))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        a, g = self.up(x).chunk(2, -1)
        return self.down(F.silu(g) * a)


class Block(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.attn = Attention(width, heads)
        self.norm2 = nn.RMSNorm(width)
        self.mlp = SwiGLU(width, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class VanillaTransformer(nn.Module):
    def __init__(self, spec, width, patch=3, octaves=9):
        super().__init__()
        assert width % HEAD_DIM == 0 and spec.view % patch == 0
        heads = width // HEAD_DIM
        depth = heads + 2
        hidden = 64 * math.ceil(8 * width / 3 / 64)
        self.spec, self.patch, self.width = spec, patch, width
        self.grid = spec.view // patch                        # patches per side
        self.n_patch = self.grid ** 2
        self.obs_tokens = (spec.steps + 1) * self.n_patch
        self.n_ctx = self.obs_tokens + spec.steps
        self.n_tokens = self.n_ctx + spec.queries

        self.patch_embed = nn.Linear(patch * patch, width)
        self.patch_pos = nn.Embedding(self.n_patch, width)
        self.step_embed = nn.Embedding(spec.steps + 1, width)
        self.dir_embed = nn.Embedding(spec.directions, width)
        self.type_embed = nn.Embedding(3, width)
        # Fourier features of the scaled offset: octaves 0..octaves-1 give
        # wavelengths 2 / 2^k in [-1, 1] units, i.e. 2 * offset_max / 2^k
        # pixels, from the full range down to about two pixels.
        self.register_buffer('freqs', math.pi * 2.0 ** torch.arange(octaves),
                             persistent=False)
        self.offset_embed = nn.Sequential(
            nn.Linear(2 * 2 * octaves, width), nn.GELU(),
            nn.Linear(width, width))

        self.blocks = nn.ModuleList(
            Block(width, heads, hidden) for _ in range(depth))
        self.norm = nn.RMSNorm(width)
        self.head = nn.Linear(width, spec.view * spec.view)

        cos, sin = _rope_cache(torch.arange(self.n_tokens), HEAD_DIM)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.apply(self._init)
        for block in self.blocks:                               # GPT-2 residual scaling
            for p in (block.attn.proj.weight, block.mlp.down.weight):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * depth))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def patches(self, obs):
        """``(B, T+1, S, S)`` -> ``(B, T+1, n_patch, patch*patch)``, patches in
        row-major order, pixels in row-major order within a patch."""
        b, t1, s, _ = obs.shape
        g, p = self.grid, self.patch
        x = obs.view(b, t1, g, p, g, p).permute(0, 1, 2, 4, 3, 5)
        return x.reshape(b, t1, g * g, p * p)

    def forward(self, obs, directions, query_offset):
        b = obs.shape[0]
        t1, q = self.spec.steps + 1, self.spec.queries
        type_vec = self.type_embed.weight

        tok = self.patch_embed(self.patches(obs))                 # (B, T+1, 9, W)
        step = self.step_embed.weight                             # (T+1, W)
        tok = tok + self.patch_pos.weight + step[:, None] + type_vec[0]
        dirs = self.dir_embed(directions) + step[:-1] + type_vec[1]  # (B, T, W)
        dirs = F.pad(dirs, (0, 0, 0, 1))                          # dummy after obs_T
        ctx = torch.cat([tok, dirs[:, :, None]], 2)               # (B, T+1, 10, W)
        ctx = ctx.flatten(1, 2)[:, :self.n_ctx]                   # drop the dummy

        off = query_offset.float() / self.spec.offset_max         # (B, Q, 2)
        ang = off[..., None] * self.freqs                          # (B, Q, 2, K)
        feats = torch.cat([ang.sin(), ang.cos()], -1).flatten(2)  # (B, Q, 4K)
        qry = self.offset_embed(feats) + type_vec[2]              # (B, Q, W)

        x = torch.cat([ctx, qry], 1)                              # (B, N, W)
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            x = block(x, cos, sin)
        out = self.head(self.norm(x[:, -q:]))                     # (B, Q, S*S)
        return out.view(b, q, self.spec.view, self.spec.view)


def build(size, spec):
    return VanillaTransformer(spec, **SIZES[size])


register('vanilla')(sys.modules[__name__])
