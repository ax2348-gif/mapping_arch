"""hybrid45b_nocopro: hybrid4.5b with coprocessor placement () (trunk blocks
run before each coprocessor: first = 0, middle = n/2, last = n; see
hybrid45b.py for the design, the muP rule and the mechanism trace).

The minimal-pair control: no coprocessor, which makes the model vanilla4
exactly (same parameters, names, initial values at a seed, and outputs;
asserted in tests/test_prefreeze.py).
"""
import sys

from mapping.models.se2d import register
from mapping.models.se2d import hybrid45b as base

SIZES = base.SIZES
LR = base.LR
SCALING_RULE = base.SCALING_RULE
PLACEMENT = ()
ARCH = {'placement': PLACEMENT}   # what this model's config file binds


def build(size, spec):
    return base.build(size, spec, placement=PLACEMENT)


register('hybrid45b_nocopro')(sys.modules[__name__])
