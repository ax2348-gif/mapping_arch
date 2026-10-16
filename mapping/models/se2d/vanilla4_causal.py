"""vanilla4_causal: vanilla4 with causal attention over the sorted token
order; the control for hybrid4_causal (separates "causal hurts" from
"causal hurts the delta rule"). Everything else is vanilla4's; the
mechanism trace is hybrid4_causal's (tests/test_causal.py).
"""
import sys

import torch.nn.functional as F

from mapping.models.se2d import register
from mapping.models.se2d.vanilla4 import (LR, SCALING_RULE, SIZES,  # noqa: F401
                                          Attention, Vanilla4, apply_rope)


class CausalAttention(Attention):
    """vanilla4's attention, masked causally over the sorted token order."""

    def forward(self, x, cos, sin):
        b, l, w = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.heads, -1).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, l, w))


class Vanilla4Causal(Vanilla4):
    def __init__(self, spec, width):
        super().__init__(spec, width)
        for blk in self.blocks:          # same modules and initial values
            blk.mixer.__class__ = CausalAttention

def build(size, spec):
    return Vanilla4Causal(spec, **SIZES[size])


register('vanilla4_causal')(sys.modules[__name__])
