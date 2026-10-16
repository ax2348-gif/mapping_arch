"""hybrid4.5b, SE(2): a fixed-size Gated DeltaNet "pose coprocessor" beside a
growing attention + SwiGLU trunk. This file registers the coprocessor-first
form; ``hybrid45b_mid``, ``hybrid45b_last``, ``hybrid45b_two`` and
``hybrid45b_nocopro`` (the minimal-pair control) reuse it with other
placements.

Design (lift notes, section 03, form "b"):

- **Trunk.** Exactly ``vanilla4`` at the rung: the same embeddings, query
  placement, RoPE on the step index, ``w/32`` blocks of softmax attention +
  SwiGLU, the same muP multipliers and initialisation. The class subclasses
  ``Vanilla4`` and builds the trunk by calling its constructor, so the
  trunk's parameters, their names and (for a given seed) their initial
  values are vanilla4's. Coprocessors are created afterwards, so they do not
  shift the trunk's random stream.
- **Coprocessor.** One Gated DeltaNet block of width 64, identical at every
  rung: ``x <- x + up(B64(down(LN(x))))``, where ``down`` maps the trunk
  width w to 64, ``B64`` is hybrid4 xs's first block (pre-norm Gated DeltaNet
  with 2 heads of 32, one per scan direction, beta in (0, 2), causal short
  convolution of width 4 in each head's scan order, then a SwiGLU MLP of
  hidden 192, with its own residual stream), and ``up`` maps 64 back to w.
  ``B64`` is built with hybrid4 xs's arguments verbatim (multiplier 4,
  std 0.5/8, residual-output std 0.5/8/2), so its parametrisation, and hence
  its training dynamics under Adam, does not depend on the rung. Its short
  convolution uses the muP-governed init of ``hybrid4_muconv`` (the decision
  of the pre-freeze item 1; docs/lift/prefreeze_ablations.md).
- **Projections, muP.** ``down`` (fan_in w grows, fan_out 64 fixed) is an
  output-like matrix: the trunk's hidden rule, multiplier 256/w and effective
  std 0.5/sqrt(w). ``up`` (fan_in 64 fixed, fan_out w grows) is input-like:
  its learning rate must not shrink with w, so it takes the coprocessor's
  own multiplier (4) and effective std 0.5/8, scaled by the trunk's
  residual-branch factor 1/sqrt(2 * trunk blocks).
- **Placement.** ``PLACEMENT`` lists, for each coprocessor, how many trunk
  blocks run before it: 0 = first, n/2 = middle, n = last (after every
  trunk block, just before the final norm and readout). Several entries give
  several independent coprocessors. An empty list is the minimal-pair
  control, which is vanilla4 exactly (asserted in tests/test_prefreeze.py:
  same state dict at the same seed, same outputs).

Parameters: one coprocessor is 59,428 (the width-64 block) + 129 w + 64
(projections, LayerNorm), 76,260 at s (w = 128) against vanilla4 s's
822,281; hybrid45b s has 898,541 (hybrid4 s: 860,505).

Mechanism trace (what has to happen for the hybrid4 mechanism to survive the
move into a coprocessor):

1. **Token geometry is unchanged.** The coprocessor runs on the sequence in
   the placed order (``token_order``: obs_t x9, queries at t, odometry t), the
   same order hybrid4's first block sees. Its scan and convolution follow
   sequence order only, so the step period (10 tokens on steps with no
   query), which patches see the odometry token through the width-4
   convolution (3 of 9 per step in the left-to-right order), and the
   right-to-left head's span are hybrid4's.
2. **The odometry symbol reaches k and beta.** At "first" the coprocessor's
   input is LN(embedding); ``down`` is a random linear map from w to 64
   dimensions. The 16 odometry embeddings stay linearly separable after it
   (16 generic vectors in 64 dimensions), so k, beta and the decay can depend
   on the symbol as in hybrid4, where the GDN input is LN(embedding) directly.
   At "middle" or "last" the input is the trunk's residual stream after
   attention, which still contains the embedding (residual) plus whatever
   attention mixed in.
3. **The rotating subspace is per head and 32-dimensional,** as in hybrid4;
   width 64 means exactly one head per direction, the configuration in which
   hybrid4 xs learned the turn-locked rotation.
4. **The read-out is positional.** The coprocessor writes its per-token
   output into each token's residual (``up``), so the pose a query token
   needs lands on that token; attention after the coprocessor can then match
   content against it. At "last" there is no attention after it: the frame
   is computed, but only the readout can use it, which is the placement
   question.

Audit: vanilla4's items for tokens, embeddings, query placement and RoPE
apply unchanged to the trunk. The coprocessor adds hybrid4's: the short
convolution and the scan mix tokens by sequence order only; beta, the decay
and the convolution are learned from token content; nothing is computed from
the odometry outside learned layers. ``down`` and ``up`` are learned linear
maps applied per token.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.models.se2d import register
from mapping.models.se2d import hybrid4_muconv as h4
from mapping.models.se2d import vanilla4 as v4

COPRO_WIDTH = 64     # fixed at every rung
SIZES = v4.SIZES
LR = dict(v4.LR)
PLACEMENT = ('first',)


def resolve(placement, n):
    """Positions (trunk blocks before each coprocessor) from names or ints."""
    names = {'first': 0, 'middle': n // 2, 'last': n}
    return tuple(sorted(names[p] if isinstance(p, str) else int(p)
                        for p in placement))


SCALING_RULE = (
    "vanilla4's ladder for the trunk (width w in 64..512, w/32 blocks of "
    'attention + SwiGLU, muP base lr 3e-4 with multiplier 256/w on hidden '
    'matrices) plus a FIXED coprocessor: one width-64 Gated DeltaNet block '
    "(hybrid4 xs's first block, 2 heads of 32, beta in (0, 2), short conv "
    'width 4, SwiGLU hidden 192), the same at every rung, entered through a '
    'learned w -> 64 projection (multiplier 256/w) and left through a 64 -> w '
    'projection (multiplier 4). Only the trunk and the two projections grow.')


class Coprocessor(nn.Module):
    """``x <- x + up(B64(down(LN(x))))`` with B64 hybrid4 xs's first block."""

    def __init__(self, width, trunk_blocks):
        super().__init__()
        c = COPRO_WIDTH
        c_mult = h4.BASE_WIDTH / c                    # 4: hybrid4 xs's
        c_std = 0.5 / math.sqrt(c)
        c_out = c_std / math.sqrt(2 * (c // h4.HEAD_DIM))   # hybrid4 xs: 2 mixers
        self.norm = nn.LayerNorm(width)
        self.down = h4.MuLinear(width, c, h4.BASE_WIDTH / width,
                                0.5 / math.sqrt(width))
        self.block = h4.Block(h4.GatedDeltaNet(c, c // h4.HEAD_DIM, c_mult,
                                               c_std, c_out),
                              c, c_mult, c_std, c_out)
        self.up = h4.MuLinear(c, width, c_mult,
                              c_std / math.sqrt(2 * trunk_blocks))

    def forward(self, x):
        h = self.block(self.down(self.norm(x)), None, None)
        return x + self.up(h)


class Hybrid45b(v4.Vanilla4):
    def __init__(self, spec, width, placement=PLACEMENT):
        super().__init__(spec, width)                 # vanilla4's trunk, exactly
        n = len(self.blocks)
        self.placement = resolve(placement, n)
        assert all(0 <= p <= n for p in self.placement), self.placement
        self.copros = nn.ModuleList(Coprocessor(width, n)
                                    for _ in self.placement)

    def forward(self, obs, odometry, query_pose, query_time=None):
        # vanilla4.forward with the coprocessors interleaved in the block loop
        directions = odometry
        query_offset, turn = query_pose[..., :2], query_pose[..., 2]
        b, t1 = obs.shape[:2]
        q = query_offset.shape[1]
        w = self.width
        dev = obs.device
        if query_time is None:
            query_time = torch.full((b, q), t1 - 1, device=dev,
                                    dtype=torch.long)
        patches = (obs.view(b, t1, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                   .reshape(b, t1, 9, 9))
        idx = torch.arange(9, device=dev)
        steps = torch.arange(t1, device=dev)
        obs_tok = self.patch_embed(patches) + self.patch_pos(idx)
        dir_tok = F.pad(self.dir_emb(directions), (0, 0, 0, 1))   # (b,33,w)
        seq = torch.cat([obs_tok, dir_tok[:, :, None]], 2)        # (b,33,10,w)
        seq = seq.reshape(b, t1 * 10, w)[:, :-1]                  # drop pad
        qry = (self.offset_embed(self.offset_features(query_offset))[:, :, None]
               + self.turn_emb(turn)[:, :, None]
               + self.query_pos(idx))                             # (b,q,9,w)
        x = torch.cat([seq, qry.reshape(b, q * 9, w)], 1)         # (b,401,w)
        ctx_pos = torch.cat([steps[:, None].expand(t1, 10)], 1).reshape(-1)[:-1]
        pos = torch.cat([ctx_pos[None].expand(b, -1),
                         query_time[:, :, None].expand(-1, -1, 9).reshape(b, -1)],
                        1)                                        # (b,401)
        order = self.token_order(query_time, t1, q)               # (b,401)
        x = torch.gather(x, 1, order[..., None].expand(-1, -1, w))
        pos = torch.gather(pos, 1, order)
        cos, sin = v4.rope_cos_sin(pos, v4.HEAD_DIM)
        for i in range(len(self.blocks) + 1):
            for p, copro in zip(self.placement, self.copros):
                if p == i:
                    x = copro(x)
            if i < len(self.blocks):
                x = self.blocks[i](x, cos, sin)
        inverse = torch.argsort(order, 1)                        # back to layout
        x = torch.gather(x, 1, inverse[..., None].expand(-1, -1, w))
        y = self.readout(self.norm(x[:, -q * 9:]))                # (b,q*9,9)
        return (y.view(b, q, 3, 3, 3, 3).permute(0, 1, 2, 4, 3, 5)
                .reshape(b, q, 9, 9))

    def gdn_layers(self):
        """``[(name, GatedDeltaNet)]`` in forward order, for the probe tools."""
        return [(f'copros.{j}.block.mixer', c.block.mixer)
                for j, c in enumerate(self.copros)]


def build(size, spec, placement=PLACEMENT):
    return Hybrid45b(spec, **SIZES[size], placement=placement)


register('hybrid45b')(sys.modules[__name__])
