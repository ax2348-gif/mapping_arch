"""vanilla4c: vanilla4 (``mapping/models/se2d/vanilla4.py``) ported to the
continuous SE(2) task: hybrid4c with every token mixer as softmax
attention with RoPE on the step index, exactly as vanilla4 is hybrid4
with every mixer as attention.

The port's changes are hybrid4c's (15 x 15 views as 25 patches of 3 x 3,
odometry and query tokens from ``mapping.se2c.encoding`` through a
Linear each); attention, SwiGLU, blocks, ladder, muP rule and init are
vanilla4's, imported from its module. There is no recurrence and no
convolution: the turn angle reaches every layer only through the
odometry token's features, and composing turns into a relative pose
must happen across attention and SwiGLU layers.

Audit: hybrid4c's items 10-13; no recurrence, no convolution.
"""
import sys

from mapping.models.se2d.vanilla4 import (LR, SCALING_RULE,  # noqa: F401
                                         SIZES, Attention, Block)
from mapping.models.se2c import register
from mapping.models.se2c.hybrid4c import Hybrid4c


class Vanilla4c(Hybrid4c):
    attention = Attention
    block = Block

    QUERY_MODES = ('mixed', 'inert')    # mapping/models/query_mode.py

    def __init__(self, spec, width, patch=3, query_mode='mixed'):
        super().__init__(spec, width, mixer='attention', patch=patch,
                         query_mode=query_mode)


def build(size, spec):
    return Vanilla4c(spec, **SIZES[size])


register('vanilla4c')(sys.modules[__name__])
