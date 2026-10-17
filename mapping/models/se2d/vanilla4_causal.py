"""vanilla4_causal: vanilla4 with causal attention over the sorted token
order; the control for hybrid4_causal (separates "causal hurts" from
"causal hurts the delta rule"). Everything else is vanilla4's; the
mechanism trace is hybrid4_causal's (tests/test_causal.py).
"""
import sys

from mapping.models.se2d import register
from mapping.models.se2d.vanilla4 import (LR, SCALING_RULE, SIZES,  # noqa: F401
                                          Attention, Vanilla4)


class CausalAttention(Attention):
    """vanilla4's attention, masked causally over the sorted token order: the
    parent's FlexAttention path with the causal mask (mapping/models/query_mode.py)."""

    causal = True


class Vanilla4Causal(Vanilla4):
    QUERY_MODES = ('mixed', 'inert')    # mapping/models/query_mode.py

    def __init__(self, spec, width, query_mode='mixed'):
        super().__init__(spec, width, query_mode)
        for blk in self.blocks:          # same modules and initial values
            blk.mixer.__class__ = CausalAttention

def build(size, spec):
    return Vanilla4Causal(spec, **SIZES[size])


register('vanilla4_causal')(sys.modules[__name__])
