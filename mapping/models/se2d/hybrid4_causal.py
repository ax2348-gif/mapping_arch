"""hybrid4_causal: hybrid4 with every operator causal in token order.

hybrid4 found its exact rotation in the RIGHT-TO-LEFT delta-rule head, and
P5 on the left-to-right state failed. Nothing in the task needs looking
ahead: queries sit right after observation t (per-step placement) and
causal coverage puts everything a query needs in its past. This model asks
whether a fully causal hybrid4 can learn the task.

Changes from hybrid4, and nothing else:

1. **Gated DeltaNet:** every head scans left to right. Same head count,
   widths, short convolution (width 4, causal in the scan direction),
   beta in (0, 2), decay, gate and norm; the right-to-left head becomes a
   second left-to-right head.
2. **Attention:** causal over the sorted token order (``is_causal``), RoPE
   on the step index as before.

Mechanism trace (asserted in tests/test_causal.py). The blocks run on
hybrid4's sorted order: per step, observation t's 9 patches, the 9 tokens
of each query asked at t, then odometry t (the move from t to t+1). Under
the causal mask a query at t therefore sees observations 0..t, odometry
0..t-1 (the moves that relate past views to pose t) and earlier queries'
tokens (query poses only, no targets), and NOT odometry t or anything
later. Within a query, patch j sees patches < j; every query token carries
the query pose.
"""
import sys

import torch
import torch.nn.functional as F

from mapping.models.se2d import register
from mapping.models.se2d.hybrid4 import (DECAY_TAU, LR, SCALING_RULE,  # noqa: F401
                                         SIZES, Attention, GatedDeltaNet,
                                         Hybrid4, apply_rope)


class CausalGatedDeltaNet(GatedDeltaNet):
    """hybrid4's Gated DeltaNet with every head scanning left to right."""

    causal = True      # read by tools/transition_probe.py (no flipped heads)

    def forward(self, x):
        b, l, _ = x.shape
        h, d = self.heads, self.d
        qkv = torch.stack([self.q(x), self.k(x), self.v(x)], 2)      # (b,l,3,h*d)
        g = (F.logsigmoid(self.decay(x).float()) / DECAY_TAU)         # (b,l,h)
        beta = 2 * torch.sigmoid(self.beta(x).float())                # (0, 2)
        seq = F.pad(qkv.reshape(b, l, 3 * h * d).transpose(1, 2), (self.conv - 1, 0))
        seq = F.silu(self.short(seq)).transpose(1, 2).view(b, l, 3, h, d)
        q, k, v = seq.unbind(2)
        q = F.normalize(q.float(), dim=-1).to(v.dtype) * d ** -0.5
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        o = self.scan(q, k, v, g, beta)                               # (b,l,h,d)
        o = self.norm(o).to(x.dtype) * F.silu(self.gate(x)).view(b, l, h, d)
        return self.out(o.reshape(b, l, h * d))


class CausalAttention(Attention):
    """hybrid4's attention, masked causally over the sorted token order."""

    def forward(self, x, cos, sin):
        b, l, w = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.heads, -1).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, l, w))


class Hybrid4Causal(Hybrid4):
    def __init__(self, spec, width):
        super().__init__(spec, width)
        # same modules and initial values as hybrid4 at the same seed; only
        # the operators' forward changes
        for blk in self.blocks:
            blk.mixer.__class__ = (CausalGatedDeltaNet
                                   if isinstance(blk.mixer, GatedDeltaNet)
                                   else CausalAttention)

def build(size, spec):
    return Hybrid4Causal(spec, **SIZES[size])


register('hybrid4_causal')(sys.modules[__name__])
