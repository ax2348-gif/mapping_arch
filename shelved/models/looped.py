"""Looped transformer: one weight-tied block iterated a fixed number of times.

Hypothesis
----------
The task is a sequence problem whose solution is an iterative computation:
summing a prefix of step tokens, matching that sum against each query's
offset embedding, then selecting and re-composing content from a handful
of observation tokens. Each stage depends on the previous one, so the
circuit has a serial depth that a fixed-depth stack can only buy with
parameters (a new block per level). The universal / looped transformer
literature (Dehghani et al. 2019; Giannou et al. 2023; Yang et al. 2024
"Looped transformers are better at learning learning algorithms") argues
the opposite trade: apply *one* block ``L`` times, so depth is a compute
knob and parameters are a width knob. This file is that arm of the study.

Tokenisation (sequence of 337 tokens, full bidirectional attention)
-------------------------------------------------------------------
* 33 x 9 patch tokens: each 9x9 observation is cut into 3x3 patches of 3x3
  pixels, ``Linear(9, d)`` + learned patch-index embedding (9 entries) +
  learned step embedding (33 entries, the observation's index in the
  sequence) + a token-type embedding.
* 32 step tokens: learned embedding of the direction symbol (4 entries) +
  the step embedding of the index ``t`` it sits at + token-type embedding.
* 8 query tokens: two learned tables over the raw offset range on each
  axis (577 entries each), summed, + token-type embedding. Queries are
  given the step index of the last observation purely as a sequence
  index: the offset is defined relative to it, and the index is the same
  constant for every query of every episode.

The block
---------
Pre-RMSNorm, multi-head attention with a learned per-head relative bias
over the *difference of step indices* (T5-style, 65 entries, no bucketing),
and a SwiGLU MLP. It is applied ``loops`` times with tied weights:

    h_0 = x0                                  # the embedded sequence
    for k in range(loops):
        c   = norm_in(x0) + e_loop[k]         # input injection + loop index
        h   = h + attn(norm1(h) + c)
        h   = h + mlp(norm2(h) + c)
    pred = readout(norm_f(h[query tokens]))

Design choices and why:
* **Loops are fixed at 12 across the ladder, not scaled.** Loops add no
  parameters, so letting them grow with the rung would put a compute
  axis inside the parameter axis of the scaling plot and the exponent
  would no longer be a parameter exponent. Holding loops fixed makes the
  sweep a pure width sweep at constant serial depth; the vanilla stack,
  which must grow depth and width together, is then the contrast. Twelve
  is the depth of a GPT-2-small stack: the serial chain here is a handful
  of dependent stages, and twelve gives two or three iterations per
  stage without making the top rung slow. ``loops`` is a build kwarg so
  a depth-at-fixed-width sweep can be run as a separate experiment.
* **Loop-index embedding** ``e_loop[k]`` (``loops x d``, negligible
  parameters): the tied block can specialise by phase (early iterations
  aggregate, late ones select and decode) instead of having to infer the
  phase from the residual stream. Keyed on the loop counter only.
* **Input injection** every loop (the "recall" of deep-thinking nets, and
  the per-layer input of the universal transformer): the normalised
  embedded input is added to each sub-layer's input, so the raw content,
  the step index and the offset embedding are available at every
  iteration and the residual stream need not spend capacity preserving
  them. It enters the sub-layer inputs, not the residual stream, so the
  stream does not grow linearly with the loop count.
* **Readout** from the query tokens after the last loop only: one linear
  layer to the 81 pixels. No intermediate supervision, because the
  harness has a single loss and the study compares final-loss scaling.

Scaling ladder and learning rate
--------------------------------
Width doubles per rung (128 ... 2048), head dimension is fixed at 64 so
heads double with width, the SwiGLU hidden is 8/3 of the width rounded
to a multiple of 64, and loops are fixed. Parameters scale as 12 d^2 plus
an O(d) embedding term, so the five rungs span ~150x.

The harness fixes one global learning rate, so learning-rate transfer is
done inside the model, muP-style, with forward multipliers: every hidden
matrix (QKV, attention output, both MLP matrices, the readout) is stored
as ``W_raw`` and used as ``W = (base_width / width) * W_raw``, with
``W_raw`` initialised so that ``W`` has the standard fan-in init. Under
Adam the update to ``W_raw`` is O(lr) per entry regardless of scale, so
the effective learning rate on hidden weights is ``lr * base_width /
width`` (the muP Adam rule), with base width 256 where the multiplier is
1 and the model is a plain transformer. Embeddings, norms, biases and the
relative-bias table keep the full learning rate, as muP prescribes for
input-like parameters. Attention logits keep the 1/sqrt(head_dim)
scaling because head_dim is constant along the ladder.

Audit: where positional or geometric information enters
--------------------------------------------------------
1. ``step_emb`` (33 learned vectors) added to patch tokens of observation
   ``t`` and to the step token at index ``t``: sequence index only.
2. ``patch_emb`` (9 learned vectors) added to each patch token: the
   patch's index within its observation, a sequence index.
3. ``rel_bias`` (heads x 65): learned attention bias keyed on the
   difference of two tokens' step indices, in -32..32. Step indices come
   from the fixed sequence layout (observation ``t`` -> ``t``, step token
   ``t`` -> ``t``, every query -> 32, the last observation's index); none
   is derived from ``directions`` or ``query_offset``.
4. ``dir_emb`` (4 learned vectors): the direction symbol is embedded as a
   token's content, like any vocabulary item. Nothing sums, scales or
   otherwise combines these embeddings outside learned layers.
5. ``off_y`` / ``off_x`` (577 learned vectors each): the raw query offset
   is looked up per axis and the two vectors are summed. The offset is
   not combined with anything else, not used to index observations, and
   never reaches the attention bias.
6. ``loop_emb`` (loops x d): keyed on the loop counter only.
7. There is no cumulative sum, no coordinate computed from the inputs, no
   canvas, no scatter or gather indexed by anything derived from the
   directions or the offset; attention is full and bidirectional over all
   337 tokens with only (3) as bias; ``positions`` is never read.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mapping.models import register

HEAD_DIM = 64
BASE_WIDTH = 256
LOOPS = 12

SIZES = {
    'xs': dict(width=128, loops=LOOPS),
    's': dict(width=256, loops=LOOPS),
    'm': dict(width=512, loops=LOOPS),
    'l': dict(width=1024, loops=LOOPS),
    'xl': dict(width=2048, loops=LOOPS, checkpoint=True),
}
# ``checkpoint`` recomputes each loop's activations in backward; it is a
# memory strategy only (the function and its gradients are unchanged) and
# is what lets the top rung train a batch of 64 in bf16 on 23 GiB.

SCALING_RULE = (
    'Width doubles per rung from 128 to 2048; head dimension is fixed at 64 '
    'so heads = width / 64 double with it; the SwiGLU hidden is 8/3 * width '
    'rounded up to a multiple of 64; one weight-tied block is iterated a '
    'fixed 12 loops at every rung, so the sweep varies width alone at '
    'constant serial depth (loops add no parameters, and scaling them would '
    'hide a compute axis inside the parameter axis). Parameters ~ 12 d^2 + '
    'O(d) embeddings. Learning rate: the harness lr (3e-4) is the base rate '
    'at width 256; every hidden matrix (QKV, attention out, both MLP '
    'matrices, readout) carries a forward multiplier 256 / width with its '
    'raw init divided by the same factor, so under Adam its effective '
    'learning rate is lr * 256 / width (muP Adam rule) while embeddings, '
    'norms and biases keep the full lr. Attention keeps 1/sqrt(64) because '
    'head_dim is constant.')


class MuLinear(nn.Module):
    """``y = x @ (mult * W_raw)^T + b``; ``W_raw`` is initialised so that the
    effective weight has std ``std``. Under Adam this gives the effective
    weight a learning rate of ``mult * lr``."""

    def __init__(self, fan_in, fan_out, mult, std=None, bias=False,
                 zero=False):
        super().__init__()
        self.mult = float(mult)
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in))
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None
        std = (1.0 / math.sqrt(fan_in)) if std is None else std
        if zero:
            nn.init.zeros_(self.weight)
        else:
            nn.init.normal_(self.weight, std=std / self.mult)

    def forward(self, x):
        return F.linear(x, self.weight * self.mult, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class Block(nn.Module):
    """One pre-norm transformer block with a relative step bias and SwiGLU."""

    def __init__(self, d, heads, hidden, mult, loops):
        super().__init__()
        self.d, self.heads = d, heads
        branch_std = 1.0 / math.sqrt(d) / math.sqrt(2 * loops)
        self.norm1 = RMSNorm(d)
        self.qkv = MuLinear(d, 3 * d, mult)
        self.proj = MuLinear(d, d, mult, std=branch_std)
        self.norm2 = RMSNorm(d)
        self.gate_up = MuLinear(d, 2 * hidden, mult)
        self.down = MuLinear(hidden, d, mult,
                             std=branch_std * math.sqrt(d / hidden))

    def attn(self, x, bias):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, HEAD_DIM).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))     # (B, H, N, hd)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))

    def mlp(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)

    def forward(self, h, c, bias):
        h = h + self.attn(self.norm1(h) + c, bias)
        h = h + self.mlp(self.norm2(h) + c)
        return h


class LoopedTransformer(nn.Module):
    def __init__(self, spec, width, loops, patch=3, base_width=BASE_WIDTH,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        if width % HEAD_DIM:
            raise ValueError(f'width {width} not a multiple of {HEAD_DIM}')
        s, t, q = spec.view, spec.steps, spec.queries
        if s % patch:
            raise ValueError(f'view {s} not divisible by patch {patch}')
        self.spec, self.d, self.loops, self.patch = spec, width, loops, patch
        self.heads = width // HEAD_DIM
        d = width
        mult = base_width / width
        hidden = 64 * math.ceil(8 * d / 3 / 64)
        g = s // patch                       # patches per side
        self.n_patch = g * g
        self.n_obs, self.n_dir, self.n_q = (t + 1) * self.n_patch, t, q
        self.ranges = (slice(0, self.n_obs),
                       slice(self.n_obs, self.n_obs + t),
                       slice(self.n_obs + t, self.n_obs + t + q))

        # token embeddings (full learning rate; std 0.02 like GPT-2 tables)
        self.patch_in = nn.Linear(patch * patch, d)
        self.patch_emb = nn.Embedding(self.n_patch, d)
        self.step_emb = nn.Embedding(t + 1, d)
        self.dir_emb = nn.Embedding(spec.directions, d)
        self.off_y = nn.Embedding(2 * spec.offset_max + 1, d)
        self.off_x = nn.Embedding(2 * spec.offset_max + 1, d)
        self.type_emb = nn.Embedding(3, d)
        self.loop_emb = nn.Parameter(torch.zeros(loops, d))
        for e in (self.patch_emb, self.step_emb, self.dir_emb, self.off_y,
                  self.off_x, self.type_emb):
            nn.init.normal_(e.weight, std=0.02)
        nn.init.normal_(self.loop_emb, std=0.02)

        # step index of every token in the fixed sequence layout
        obs_steps = torch.arange(t + 1).repeat_interleave(self.n_patch)
        dir_steps = torch.arange(t)
        q_steps = torch.full((q,), t, dtype=torch.long)
        steps = torch.cat([obs_steps, dir_steps, q_steps])
        self.register_buffer('rel_index', steps[:, None] - steps[None, :] + t,
                             persistent=False)
        self.rel_bias = nn.Parameter(torch.zeros(self.heads, 2 * t + 1))

        self.norm_in = RMSNorm(d)
        self.block = Block(d, self.heads, hidden, mult, loops)
        self.norm_f = RMSNorm(d)
        self.readout = MuLinear(d, s * s, mult, zero=True, bias=True)

    def embed(self, obs, directions, query_offset):
        b = obs.shape[0]
        s, p, g = self.spec.view, self.patch, self.spec.view // self.patch
        t = self.spec.steps
        # (B, T+1, S, S) -> (B, T+1, g, g, p, p) -> (B, (T+1)*g*g, p*p)
        patches = (obs.view(b, t + 1, g, p, g, p).permute(0, 1, 2, 4, 3, 5)
                   .reshape(b, (t + 1) * g * g, p * p))
        pidx = torch.arange(g * g, device=obs.device).repeat(t + 1)
        sidx = torch.arange(t + 1, device=obs.device).repeat_interleave(g * g)
        x_obs = (self.patch_in(patches) + self.patch_emb(pidx)
                 + self.step_emb(sidx) + self.type_emb.weight[0])
        x_dir = (self.dir_emb(directions)
                 + self.step_emb(torch.arange(t, device=obs.device))
                 + self.type_emb.weight[1])
        off = query_offset + self.spec.offset_max
        x_q = (self.off_y(off[..., 0]) + self.off_x(off[..., 1])
               + self.type_emb.weight[2])
        return torch.cat([x_obs, x_dir, x_q], 1)

    def forward(self, obs, directions, query_offset):
        b, q = query_offset.shape[:2]
        s = self.spec.view
        x0 = self.embed(obs, directions, query_offset)
        bias = self.rel_bias[:, self.rel_index][None]        # (1, H, N, N)
        bias = bias.to(x0.dtype if not torch.is_autocast_enabled()
                       else torch.get_autocast_dtype('cuda'))
        c0 = self.norm_in(x0)
        h = x0
        ckpt = self.checkpoint and self.training and torch.is_grad_enabled()
        for k in range(self.loops):
            c = c0 + self.loop_emb[k]
            if ckpt:
                h = checkpoint(self.block, h, c, bias, use_reentrant=False)
            else:
                h = self.block(h, c, bias)
        out = self.readout(self.norm_f(h[:, self.ranges[2]]))
        return out.view(b, q, s, s)


def build(size, spec):
    return LoopedTransformer(spec, **SIZES[size])


register('looped')(sys.modules[__name__])
