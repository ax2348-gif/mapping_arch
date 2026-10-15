"""Slot-memory transformer: a token stream plus a bank of learned latent tokens.

Hypothesis under test. A vanilla transformer has no persistent state apart
from the per-token residual streams, each tied to one sequence position,
so any global representation of the episode has to be spread across those
streams and re-assembled by every query. This variant adds K learned
latent tokens ("slots") outside the input sequence: a Perceiver /
register-token / slot-attention style latent bottleneck. The latents are
written by cross-attention from the observation and direction tokens,
refined by self-attention among themselves, and read back into the token
stream, over `depth` write-read rounds. Queries never see the input tokens:
they cross-attend to the final latents only, conditioned on an embedding
of the raw query offset. The latents are therefore a learned bottleneck
through which all episode information must pass, and what each latent
holds, and which latents a query attends to, are learned from the data
with no layout or addressing scheme imposed by hand.

Architecture, for width d, heads h, depth L (rounds), K latents:

    tokens  o_0 d_0 o_1 d_1 ... d_31 o_32         65 tokens of width d
            o_t = Linear(81 -> d)(window_t)      d_t = Embedding(4)(dir_t)
            + learned type embedding (obs / dir) + learned sequence-index
            embedding (position in the 65-token sequence)
    latents S in R^{K x d}, a learned parameter, same init for every episode

    round r = 1..L:
        tokens:  x = x + SelfAttn(LN x);  x = x + MLP(LN x)
        write:   s = s + CrossAttn(q = LN s, kv = LN x)
        refine:  s = s + SelfAttn(LN s);  s = s + MLP(LN s)
        read:    x = x + CrossAttn(q = LN x, kv = LN s)     (not after round L)

    query   q = MLP(Fourier(offset / offset_max))             one token per query
    decode, L_dec layers:  q = q + CrossAttn(q = LN q, kv = LN s);  q = q + MLP(LN q)
    head    Linear(LN q -> 81) -> (9, 9) window

Design choices and why:

* Write *and* read every round (interleaved cross-attention in both
  directions), rather than a single Perceiver-style write followed by
  latent-only self-attention. What an observation token should contribute
  depends on direction tokens far from it in the sequence, and that
  dependency is itself something the network has to compute. The read
  path lets every token condition on the current latent state, so the
  aggregation can be refined over rounds instead of having to be resolved
  in one attention pass. The latents are updated by residual addition
  across rounds (iterative refinement without a GRU, so every update is a
  plain pre-LN transformer block).
* Latent self-attention and MLP every round, as in the Perceiver latent
  transformer, so the latents can exchange and de-duplicate what they
  hold.
* Queries read only the latents. If the queries could also attend to the
  input tokens the bottleneck would be optional and the hypothesis would
  not be tested. K is at least the number of observation tokens at every
  rung so the bottleneck is not narrower than the input at the bottom of
  the ladder.
* Query offset enters through log-spaced Fourier features (NeRF-style, 9
  octaves, the finest with a period of about 2 px) and an MLP: a standard
  continuous-input embedding that gives neighbouring offsets related
  embeddings, instead of a 577-entry table per axis.
* Pre-LN, GPT-2 initialisation with residual-branch projections scaled by
  1 / sqrt(2 * branches), bf16-safe (no custom kernels; attention via
  F.scaled_dot_product_attention).

Scaling: see SCALING_RULE. Everything is derived from the head count h.

Audit: every place positional or geometric information enters the network
-------------------------------------------------------------------------
1. Sequence-index embedding `pos` (65 x d), indexed by the token's position
   in the interleaved [o_0, d_0, ..., d_31, o_32] sequence. Keyed on the
   sequence index only, never on directions or anything derived from
   them. Allowed ("learned per-step embeddings").
2. Type embedding `kind` (2 x d): says whether a token is an observation
   or a direction. Not positional.
3. Direction embedding `dir_emb` (4 x d): a learned embedding of the raw
   direction symbol at each step. The directions are never summed,
   cumulated, or converted to coordinates outside the learned layers; the
   only way they can be integrated is through attention and MLPs.
4. Query offset: `query_offset / offset_max` (a fixed constant from the
   TaskSpec, used only to put the raw input in [-1, 1]) -> fixed Fourier
   features -> learned MLP. The offset is embedded on its own and is never
   combined arithmetically with anything derived from the directions or
   observations; the only place it meets the rest of the network is as
   the query of a learned cross-attention over the latents.
5. Latents `slot_init` (K x d): learned free parameters, identical for
   every episode, with no spatial layout. Nothing is scattered onto or
   gathered from them by coordinate; they are written only by learned
   cross-attention whose keys come from the tokens and read only by
   learned cross-attention whose queries come from the tokens or the
   offset embedding.
6. No RoPE / relative bias at all; attention is plain dot-product with no
   positional term. No positions tensor is read; forward() takes exactly
   (obs, directions, query_offset).
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.models import register

HEAD_DIM = 32
FOURIER_OCTAVES = 9


def _rung(h):
    """The whole rung from its head count h."""
    return dict(heads=h, width=HEAD_DIM * h, depth=h, slots=16 * h,
                dec_depth=max(1, h // 2))


SIZES = {
    'xs': _rung(2),
    's': _rung(3),
    'm': _rung(4),
    'l': _rung(6),
    'xl': _rung(8),
    '2xl': _rung(12),
    '3xl': _rung(16),
}

SCALING_RULE = (
    'One-parameter family in the head count h = 2, 3, 4, 6, 8, 12, 16 '
    '(xs .. 3xl): width = 32 h (head dim fixed at 32), encoder rounds '
    '(depth) = h, slots = 16 h, slot dimension = width, decoder depth = '
    'max(1, h // 2), MLP hidden = 4 x width throughout. Parameters go as '
    '~ (32 depth + 12 dec_depth) width^2, i.e. ~ h^3. Learning rate: the '
    'harness default 3e-4 (AdamW, betas 0.9/0.95, wd 0.1, cosine) is used '
    'unchanged at every rung; widths 64-512 with pre-LN and 1/sqrt(2 '
    'branches) residual init are in the regime where a constant Adam LR of '
    '3e-4 is standard (GPT-2 small to medium), so no per-rung rule is '
    'applied. If a rung diverges, the stated fallback is lr = 3e-4 * '
    '(128 / width) for the hidden matrices (muP-style), passed via --lr.')


class Attention(nn.Module):
    """Multi-head attention with separate query and key/value sources.

    `forward(x, ctx)`: queries from `x` (B, N, d), keys/values from `ctx`
    (B, M, d). Self-attention is `forward(x, x)`. No positional term.
    """

    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x, ctx):
        b, n, d = x.shape
        m = ctx.shape[1]
        h, hd = self.heads, d // self.heads
        q = self.q(x).view(b, n, h, hd).transpose(1, 2)
        k, v = self.kv(ctx).view(b, m, 2, h, hd).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, 4 * width)
        self.proj = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Round(nn.Module):
    """One write-read round: token block, slot write, slot refine, token read."""

    def __init__(self, width, heads, read=True):
        super().__init__()
        self.tok_ln1 = nn.LayerNorm(width)
        self.tok_attn = Attention(width, heads)
        self.tok_ln2 = nn.LayerNorm(width)
        self.tok_mlp = MLP(width)

        self.write_ln_s = nn.LayerNorm(width)
        self.write_ln_x = nn.LayerNorm(width)
        self.write = Attention(width, heads)

        self.slot_ln1 = nn.LayerNorm(width)
        self.slot_attn = Attention(width, heads)
        self.slot_ln2 = nn.LayerNorm(width)
        self.slot_mlp = MLP(width)

        self.has_read = read
        if read:
            self.read_ln_x = nn.LayerNorm(width)
            self.read_ln_s = nn.LayerNorm(width)
            self.read = Attention(width, heads)

    def forward(self, x, s):
        xn = self.tok_ln1(x)
        x = x + self.tok_attn(xn, xn)
        x = x + self.tok_mlp(self.tok_ln2(x))

        s = s + self.write(self.write_ln_s(s), self.write_ln_x(x))
        sn = self.slot_ln1(s)
        s = s + self.slot_attn(sn, sn)
        s = s + self.slot_mlp(self.slot_ln2(s))

        if self.has_read:
            x = x + self.read(self.read_ln_x(x), self.read_ln_s(s))
        return x, s


class DecoderLayer(nn.Module):
    """Query tokens read the slots; queries do not attend to each other."""

    def __init__(self, width, heads):
        super().__init__()
        self.ln_q = nn.LayerNorm(width)
        self.ln_s = nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = MLP(width)

    def forward(self, q, s):
        q = q + self.attn(self.ln_q(q), self.ln_s(s))
        q = q + self.mlp(self.ln2(q))
        return q


class SlotMemoryTransformer(nn.Module):
    def __init__(self, spec, width, heads, depth, slots, dec_depth):
        super().__init__()
        self.spec = spec
        s, t = spec.view, spec.steps
        self.n_tokens = 2 * t + 1

        # token embeddings (audit items 1-3)
        self.obs_proj = nn.Linear(s * s, width)
        self.dir_emb = nn.Embedding(spec.directions, width)
        self.kind = nn.Parameter(torch.zeros(2, width))
        self.pos = nn.Parameter(torch.zeros(self.n_tokens, width))

        # latent memory (audit item 5)
        self.slot_init = nn.Parameter(torch.zeros(slots, width))

        self.rounds = nn.ModuleList(
            Round(width, heads, read=i < depth - 1) for i in range(depth))

        # query embedding (audit item 4)
        self.register_buffer(
            'freqs', math.pi * 2.0 ** torch.arange(FOURIER_OCTAVES).float(),
            persistent=False)
        n_feat = 2 * (1 + 2 * FOURIER_OCTAVES)
        self.query_mlp = nn.Sequential(nn.Linear(n_feat, width), nn.GELU(),
                                       nn.Linear(width, width))
        self.decoder = nn.ModuleList(
            DecoderLayer(width, heads) for _ in range(dec_depth))
        self.ln_out = nn.LayerNorm(width)
        self.head = nn.Linear(width, s * s)

        self._init_weights(depth, dec_depth)

    def _init_weights(self, depth, dec_depth):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.kind, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.slot_init, std=0.02)
        # residual-branch output projections: 3 branches per stream per
        # round in the encoder, 2 per decoder layer
        for r in self.rounds:
            for mod in r.modules():
                if isinstance(mod, (Attention, MLP)):
                    nn.init.normal_(mod.proj.weight,
                                    std=0.02 / math.sqrt(2 * 3 * depth))
        for layer in self.decoder:
            for mod in layer.modules():
                if isinstance(mod, (Attention, MLP)):
                    nn.init.normal_(mod.proj.weight,
                                    std=0.02 / math.sqrt(2 * 2 * dec_depth))

    def embed_tokens(self, obs, directions):
        b = obs.shape[0]
        o = self.obs_proj(obs.flatten(2)) + self.kind[0]          # (B, T+1, d)
        d = self.dir_emb(directions) + self.kind[1]               # (B, T, d)
        x = torch.empty(b, self.n_tokens, o.shape[-1], dtype=o.dtype,
                        device=o.device)
        x[:, 0::2] = o
        x[:, 1::2] = d
        return x + self.pos

    def embed_queries(self, query_offset):
        x = query_offset.float() / self.spec.offset_max             # (B, Q, 2)
        ang = x[..., None] * self.freqs                             # (B, Q, 2, F)
        feats = torch.cat([x[..., None], ang.sin(), ang.cos()], -1)
        return self.query_mlp(feats.flatten(2))

    def forward(self, obs, directions, query_offset):
        b, q = query_offset.shape[:2]
        s = self.spec.view
        x = self.embed_tokens(obs, directions)
        slots = self.slot_init.unsqueeze(0).expand(b, -1, -1)
        for r in self.rounds:
            x, slots = r(x, slots)
        y = self.embed_queries(query_offset)
        for layer in self.decoder:
            y = layer(y, slots)
        return self.head(self.ln_out(y)).view(b, q, s, s)


def build(size, spec):
    return SlotMemoryTransformer(spec, **SIZES[size])


register('slots')(sys.modules[__name__])
