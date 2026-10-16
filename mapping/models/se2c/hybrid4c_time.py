"""hybrid4c_time: hybrid4c with the recurrence restricted to the time axis.

In hybrid4c every Gated DeltaNet layer scans ALL tokens: each view's 25
patches flattened in raster order, the odometry tokens and the query
patches. The patch order within a step is arbitrary, and the delta rule's
updates do not commute, so the state depends on it. Here the recurrence
sees only tokens that live on the time axis; patches are handled as sets
by attention (RoPE on the step index, as before).

Layers (same count and alternation as hybrid4c; at s: 4):

    [time scan, attention, time scan, LAST]

``LAST`` is the ablation switch (the user's): attention, exactly hybrid4's
(``hybrid4c_time``), or a full bidirectional Gated DeltaNet over ALL
tokens in hybrid4c's token order, exactly hybrid4c's layer
(``hybrid4c_time_full``, in its own module). The switch swaps one
layer's operator; the layer count is the same.

The time scan (``mapping.models.se2d.hybrid4_time.TimeScan``), per block:

1. **Time tokens.** For each step t, in the order of hybrid4c's layout:
   a reader R_t, then one token per query asked at t, then the odometry
   token O_t (none for the last step). R_t is the mean of observation t's
   patch tokens; a query token is the mean of that query's 25 tokens;
   O_t is the odometry token itself (all from the block's normalised
   input). 33 + 8 + 32 = 73 tokens per episode against hybrid4c's 1057.
2. **Only odometry tokens write.** Readers and query tokens are read-only:
   beta = 0 and no decay (g = 0), so they cannot change the state; the
   state holds composed motion only (a pose integrator).
3. **One rotation per odometry token.** FLA's gated DeltaProduct with 2
   Householder steps per token: O_t gives two keys, two values and two
   betas, each beta in (0, 2), so its transition
   (I - b2 k2 k2^T)(I - b1 k1 k1^T) can be a rotation (two reflections,
   det +1). hybrid4 had to get the second reflection from the patches
   after the odometry token through the short convolution; there is no
   convolution here.
4. **Both directions**, as hybrid4c: heads [0, H/2) scan left to right,
   [H/2, H) right to left (the time sequence reversed).
5. **Injection.** The scan output (per-head RMSNorm, SiLU gate, output
   projection, as hybrid4c's layer) at R_t is added to every patch token
   of observation t, at a query token to every token of that query, and
   at O_t to the odometry token: the block's residual update.

The main sequence is kept in layout order [observations, odometry,
queries]; attention is permutation-equivariant apart from RoPE on the
step index, so token order matters only to the full GDN layer, which
gathers into hybrid4c's order (``token_order``) and back.

Audit: hybrid4c's items 10-13; patches enter the recurrence only through
the pooled, read-only reader and query tokens, never as a sequence.
"""
import sys

import torch

import mapping.models.se2c.hybrid4c as h4c
from mapping.models.se2d.hybrid4 import (HEAD_DIM, LR, SCALING_RULE,  # noqa: F401
                                        SIZES)
from mapping.models.se2d.hybrid4_time import layer_blocks, time_context
from mapping.models.se2c import register
from mapping.models.se2c.hybrid4c import (Hybrid4c, patchify, token_order,
                                          unpatchify)


class Hybrid4cTime(Hybrid4c):
    """hybrid4c's tokeniser, embeddings and readout (inherited); the time
    scan blocks of ``mapping.models.se2d.hybrid4_time``; the main sequence in
    layout order [observations, odometry, queries] (``sequence``).
    The encodings and RoPE table are hybrid4c's, called through its module
    so the structural tests' spies see them."""

    sequence = 'layout'      # main-sequence order the blocks receive

    def __init__(self, spec, width, last='attention', patch=3):
        super().__init__(spec, width, patch=patch)
        self.last = last
        self.blocks = layer_blocks(width, last)

    def forward(self, obs, odometry, query_pose, query_time=None):
        b, t1 = obs.shape[:2]
        q = query_pose.shape[1]
        w, P, S = self.width, self.P, self.spec.view
        dev = obs.device
        if query_time is None:
            query_time = torch.full((b, q), t1 - 1, device=dev, dtype=torch.long)
        idx = torch.arange(P, device=dev)
        steps = torch.arange(t1, device=dev)
        obs_tok = (self.patch_embed(patchify(obs, self.patch))
                   + self.patch_pos(idx)).reshape(b, t1 * P, w)
        odo_tok = self.odo_embed(h4c.encode_odometry(odometry[..., :2], odometry[..., 2]))
        qry = (self.query_embed(h4c.encode_query(query_pose[..., :2],
                                             query_pose[..., 2]))[:, :, None]
               + self.query_pos(idx)).reshape(b, q * P, w)
        x = torch.cat([obs_tok, odo_tok, qry], 1)
        n_obs, n_ctx = t1 * P, t1 * P + t1 - 1
        pos = torch.cat([steps[:, None].expand(t1, P).reshape(1, -1).expand(b, -1),
                         steps[None, :-1].expand(b, -1),
                         query_time[:, :, None].expand(-1, -1, P).reshape(b, -1)], 1)
        cos, sin = h4c.rope_cos_sin(pos, HEAD_DIM)
        obs_idx = torch.arange(n_obs, device=dev).view(t1, P)
        odo_idx = torch.arange(n_obs, n_ctx, device=dev)
        qry_idx = n_ctx + torch.arange(q * P, device=dev).view(q, P)
        ctx = time_context(obs_idx, odo_idx, qry_idx, query_time, x.shape[1])
        if self.last == 'gdn':      # hybrid4c's token order, from this layout
            here = torch.cat([obs_idx, torch.cat([odo_idx, odo_idx.new_full((1,), -1)])[:, None]],
                             1).reshape(-1)[:-1]
            here = torch.cat([here, qry_idx.reshape(-1)])
            order = torch.gather(here[None].expand(b, -1), 1,
                                 token_order(query_time, t1, q, P))
            ctx['order'], ctx['inverse'] = order, torch.argsort(order, 1)
        for block in self.blocks:
            x = block(x, cos, sin, ctx)
        y = self.readout(self.norm(x[:, -q * P:]))
        return unpatchify(y.view(b, q, P, self.patch ** 2), S, self.patch)


def build(size, spec):
    return Hybrid4cTime(spec, **SIZES[size])


register('hybrid4c_time')(sys.modules[__name__])
