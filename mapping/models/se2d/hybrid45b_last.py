"""hybrid45b_last: hybrid4.5b with coprocessor placement ('last',) (trunk blocks
run before each coprocessor: first = 0, middle = n/2, last = n; see
hybrid45b.py for the design, the muP rule and the mechanism trace).
"""
import sys

from mapping.models.se2d import register
from mapping.models.se2d import hybrid45b as base

SIZES = base.SIZES
LR = base.LR
SCALING_RULE = base.SCALING_RULE
PLACEMENT = ('last',)


def build(size, spec):
    return base.build(size, spec, placement=PLACEMENT)


register('hybrid45b_last')(sys.modules[__name__])
