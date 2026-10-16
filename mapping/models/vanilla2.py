"""The second vanilla arm: the same 2026-default transformer block, the
other standard option at every interface fork.

``vanilla`` and ``vanilla2`` are deliberate counterparts. Both are faithful
general-purpose transformers (pre-norm RMSNorm, multi-head attention with
QK-norm, SwiGLU, no biases, no dropout, GPT-2 init) and neither knows the
task is about space. They differ only in how the task's inputs are turned
into tokens and read out, which is where an architect's taste could change
the scaling outcome. ``vanilla`` took one standard option at each fork;
this file takes the other, so the pair brackets the design space rather
than reporting one draw from it.

Fork by fork (``vanilla`` -> ``vanilla2``):

1. Tokenisation: nine 3x3 patch tokens per window -> ONE token per window.
   The 81 pixels of a 9x9 observation go through a single linear embed
   (a ViT whose patch size equals the image; the "frame token" option).
   The sequence is 33 observation tokens, nothing else.
2. Direction placement: a separate direction token between windows -> the
   direction is FUSED into the observation token it leads to. Step
   direction t (from window t to window t + 1) is added to token t + 1 as
   a learned embedding; token 0 has no incoming move and gets a learned
   "no move" entry (index 4 of a 5-entry table). Additive rather than
   concatenated because, under a linear embed, concatenating a one-hot
   direction with the pixels is exactly equivalent to adding a learned
   per-direction vector after the embed (``W [x; e_d] = W_x x + w_d``),
   so the additive form is the same model written as the BERT segment /
   ViT class-embedding pattern, with the equivalence explicit.
3. Positional encoding: RoPE on the sequence index -> LEARNED ABSOLUTE
   per-step embeddings (BERT / ViT style), a 33-entry table added to the
   token of step t. Keyed on the step index only.
4. Readout: queries as extra tokens in the encoder (MAE / [MASK]) -> a
   CROSS-ATTENTION DECODER (Perceiver IO / DETR / T5 pattern). Query
   tokens carry only an embedding of their raw offset; a short decoder of
   pre-norm blocks [self-attention among the queries -> cross-attention
   from queries to the encoder output -> SwiGLU] reads the memory, then
   final RMSNorm and a linear head to the 81 pixels. Queries never enter
   the encoder. Self-attention among queries is kept because that is the
   standard decoder block (T5, DETR); the queries are independent targets,
   so it is expected to matter little.
5. Offset encoding: Fourier features + MLP -> a LEARNED TABLE over the
   integer offset range, one 577-entry table per axis (dy and dx in
   [-offset_max, offset_max]), summed. Summed rather than concatenated:
   factorised 2-D position tables are summed in the vision literature
   (axis-wise ViT / LayoutLM-style position embeddings) and the sum keeps
   the query at the model width with no extra projection.
6. Ladder: head dim 64 and width as the one parameter in both arms, but
   with one token per step the sequence is 10x shorter, so this arm adds
   a depth floor (depth = max(4, width / 64)) and runs widths 64..768 so
   the parameter counts, not the compute, match the first arm's ladder.
   Decoder depth is fixed at 2 (the shallow Perceiver IO readout; T5
   would scale it with the encoder). See SCALING_RULE.
7. Learning rate: single 3e-4 at every rung -> a per-rung rule,
   lr = 3e-4 * 256 / width, in the module-level ``LR`` dict that train.py
   honours when --lr is not given (the muP hidden-matrix rule, lr inverse
   in width, applied to every parameter group; it equals the harness
   default at width 256).

Token layout (33 encoder tokens, 8 decoder queries, fixed):

    enc: [obs_0 + nomove + pos_0][obs_1 + dir_0 + pos_1] ... [obs_32 + dir_31 + pos_32]
    dec: [off_y[dy_0] + off_x[dx_0]] ... [off_y[dy_7] + off_x[dx_7]]

Audit: where positional or geometric information enters, and why each is
within the rules.

1. ``step_pos`` (learned, 33 entries): the index t of an observation in
   the episode, added to its token. Pure sequence index, the same table
   for every episode; it carries nothing about where the window is.
2. ``dir_embed`` (learned, 5 entries: four directions plus "no move"):
   the direction index of the move that led to the observation, added to
   that observation's token as content. The directions are never summed,
   indexed with, or turned into coordinates outside learned layers; the
   encoder has to integrate them itself. Gathering ``directions[t - 1]``
   for token t is a fixed shift of the input sequence by one, not a
   computation on the direction values.
3. ``off_y`` / ``off_x`` (learned, 577 entries each): the raw query offset,
   used only as a lookup key into a learned table and summed into the
   content of the query token (the contract's "learned table over the
   offset range"). It never keys an attention bias, a gather, or any
   arithmetic with the directions, the observations, or a position.
4. Cross-attention in the decoder: queries attend to the 33 encoder
   outputs with no positional bias of any kind; which step a query reads
   from is decided entirely by learned content matching.
5. The 9x9 window is flattened row-major into 81 inputs for the linear
   embed; a fixed reshape applied identically to every observation, no
   coordinates computed, nothing placed by position.

No RoPE, no relative bias, no ALiBi anywhere. Nothing reads ``positions``.
No cumulative sums, no canvas, no scatter or gather at computed
coordinates.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.models import register

HEAD_DIM = 64
MIN_DEPTH = 4
DEC_DEPTH = 2

from mapping.models import ladder  # noqa: E402
SIZES = ladder('vanilla2')   # conf/ladders/vanilla2.yaml

SCALING_RULE = (
    'One-parameter family in width: head dim is fixed at 64, heads = '
    'width / 64, encoder depth = max(4, width / 64) (so 64/1/4, 128/2/4, '
    '256/4/4, 384/6/6, 512/8/8, 768/12/12 for width/heads/depth), decoder '
    'depth fixed at 2 blocks of [self-attn, cross-attn, SwiGLU], SwiGLU '
    'hidden = 8/3 width rounded up to a multiple of 64 in both. Below '
    'width 256 the depth floor makes parameters grow about as width^2, '
    'above it as width^3; the six widths 64..768 give 0.4M..105M '
    'parameters, matching the first arm\'s ladder in parameters (one '
    'token per step makes every rung about 10x cheaper per step than '
    'vanilla). Learning rate: per-rung rule LR[size] = 3e-4 * 256 / width '
    '(1.2e-3, 6e-4, 3e-4, 2e-4, 1.5e-4, 1e-4), the muP hidden-matrix rule '
    'of lr inverse in width applied to every parameter, anchored at the '
    'harness default 3e-4 at width 256; train.py uses it when --lr is '
    'not given. Residual-branch output projections are initialised with '
    'std 0.02 / sqrt(2 depth) in the encoder and 0.02 / sqrt(3 '
    'DEC_DEPTH) in the decoder, else std 0.02.')

LR = {size: 3e-4 * 256 / kw['width'] for size, kw in SIZES.items()}


class Attention(nn.Module):
    """Multi-head attention with QK-norm; self-attention when ``ctx`` is
    None, cross-attention from ``x`` to ``ctx`` otherwise."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.kv = nn.Linear(width, 2 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, ctx=None):
        b, n, w = x.shape
        src = x if ctx is None else ctx
        q = self.q(x).view(b, n, self.heads, self.head_dim)
        k, v = self.kv(src).view(b, -1, 2, self.heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # (B, H, N, D)
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


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.attn = Attention(width, heads)
        self.norm2 = nn.RMSNorm(width)
        self.mlp = SwiGLU(width, hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    """Pre-norm [self-attention among queries, cross-attention to the
    encoder memory, SwiGLU], the T5 / DETR decoder block."""

    def __init__(self, width, heads, hidden):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.self_attn = Attention(width, heads)
        self.norm2 = nn.RMSNorm(width)
        self.cross_attn = Attention(width, heads)
        self.norm3 = nn.RMSNorm(width)
        self.mlp = SwiGLU(width, hidden)

    def forward(self, x, mem):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), mem)
        return x + self.mlp(self.norm3(x))


class VanillaTransformer2(nn.Module):
    def __init__(self, spec, width):
        super().__init__()
        assert width % HEAD_DIM == 0
        heads = width // HEAD_DIM
        depth = max(MIN_DEPTH, heads)
        hidden = 64 * math.ceil(8 * width / 3 / 64)
        self.spec, self.width = spec, width
        t1 = spec.steps + 1
        self.n_offsets = 2 * spec.offset_max + 1

        self.obs_embed = nn.Linear(spec.view * spec.view, width)
        self.dir_embed = nn.Embedding(spec.directions + 1, width)  # + "no move"
        self.step_pos = nn.Embedding(t1, width)
        self.off_y = nn.Embedding(self.n_offsets, width)
        self.off_x = nn.Embedding(self.n_offsets, width)

        self.encoder = nn.ModuleList(
            EncoderBlock(width, heads, hidden) for _ in range(depth))
        self.enc_norm = nn.RMSNorm(width)
        self.decoder = nn.ModuleList(
            DecoderBlock(width, heads, hidden) for _ in range(DEC_DEPTH))
        self.dec_norm = nn.RMSNorm(width)
        self.head = nn.Linear(width, spec.view * spec.view)

        self.apply(self._init)
        for block in self.encoder:                              # GPT-2 residual scaling
            for p in (block.attn.proj.weight, block.mlp.down.weight):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * depth))
        for block in self.decoder:
            for p in (block.self_attn.proj.weight, block.cross_attn.proj.weight,
                      block.mlp.down.weight):
                nn.init.normal_(p, std=0.02 / math.sqrt(3 * DEC_DEPTH))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs, directions, query_offset):
        b = obs.shape[0]
        q, s = self.spec.queries, self.spec.view

        # incoming move of each observation: "no move" for obs_0, then the
        # direction of the step that led to obs_t (a shift by one, no sums)
        nomove = directions.new_full((b, 1), self.spec.directions)
        incoming = torch.cat([nomove, directions], 1)              # (B, T+1)
        x = (self.obs_embed(obs.flatten(2)) + self.dir_embed(incoming)
             + self.step_pos.weight)                                # (B, T+1, W)
        for block in self.encoder:
            x = block(x)
        mem = self.enc_norm(x)

        idx = (query_offset + self.spec.offset_max).clamp_(0, self.n_offsets - 1)
        y = self.off_y(idx[..., 0]) + self.off_x(idx[..., 1])      # (B, Q, W)
        for block in self.decoder:
            y = block(y, mem)
        out = self.head(self.dec_norm(y))                           # (B, Q, S*S)
        return out.view(b, q, s, s)


def build(size, spec):
    return VanillaTransformer2(spec, **SIZES[size])


register('vanilla2')(sys.modules[__name__])
