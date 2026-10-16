"""hybrid4_mix: hybrid4 with the causal / bidirectional choice of every
layer exposed as architecture fields (the causal interrogation, A and B).

Fields (``Hybrid4Mix(spec, width, delta, attn, top, top_op)``):

- ``delta``: direction of the TRUNK's Gated DeltaNet layers: ``'both'``
  (hybrid4: half the heads left to right, half right to left), ``'l2r'``
  (every head left to right: causal) or ``'r2l'`` (every head right to
  left: anti-causal; a diagnostic).
- ``attn``: the TRUNK's attention, ``'bidir'`` (hybrid4) or ``'causal'``
  (masked over the sorted token order).
- ``top``: the number k of FINAL layers that are fully bidirectional
  (delta rule ``'both'``, attention ``'bidir'``); the layers below are the
  trunk. Layers are hybrid4's blocks in order (xs: [GDN, attn]; s: [GDN,
  attn, GDN, attn]); k counts blocks from the top.
- ``top_op``: the operator of the top k layers: ``'native'`` (keep the
  alternation), ``'gdn'`` or ``'attn'`` (every top layer that operator,
  bidirectional). Swapping an operator changes the parameter count.

Corners: (both, bidir, 0, native) is hybrid4; (l2r, causal, 0, native) is
hybrid4_causal; tests assert both equalities (same initial values,
identical outputs). Every other variant is registered below as its own
model name with its fields fixed (``VARIANTS``), one config file each in
conf/model/se2d/.

Everything else is hybrid4's: modules are built by hybrid4's constructor at
the same seed, then each mixer's forward is switched by class, so initial
values match hybrid4's wherever the operator is unchanged.
"""
import math
import sys
import types

import torch

from mapping.models.se2d import register
from mapping.models.se2d.hybrid4 import (BASE_WIDTH, HEAD_DIM, LR,  # noqa: F401
                                         SCALING_RULE, SIZES, Attention, Block,
                                         GatedDeltaNet, Hybrid4)
from mapping.models.se2d.hybrid4_causal import (CausalAttention,
                                                CausalGatedDeltaNet)

DELTA = ('both', 'l2r', 'r2l')
ATTN = ('bidir', 'causal')
TOP_OP = ('native', 'gdn', 'attn')


class ReverseGatedDeltaNet(GatedDeltaNet):
    """Every head scans right to left: the causal layer on the reversed
    sequence (its short convolution is causal in that reversed order, as
    hybrid4's right-to-left heads)."""

    scan_dirs = 'r2l'      # read by tools/transition_probe.py

    def forward(self, x):
        return CausalGatedDeltaNet.forward(self, x.flip(1)).flip(1)


def _gdn_class(direction):
    return {'both': GatedDeltaNet, 'l2r': CausalGatedDeltaNet,
            'r2l': ReverseGatedDeltaNet}[direction]


class Hybrid4Mix(Hybrid4):
    def __init__(self, spec, width, delta='both', attn='bidir', top=0,
                 top_op='native'):
        super().__init__(spec, width)
        assert delta in DELTA and attn in ATTN and top_op in TOP_OP
        n = len(self.blocks)
        assert 0 <= top <= n, f'top={top} but the model has {n} layers'
        self.delta, self.attn, self.top, self.top_op = delta, attn, top, top_op
        heads = width // HEAD_DIM
        mult = BASE_WIDTH / width
        std = 0.5 / math.sqrt(width)
        out_std = std / math.sqrt(2 * n)
        for i, blk in enumerate(self.blocks):
            is_top = i >= n - top
            if is_top and top_op != 'native':
                want_gdn = top_op == 'gdn'
                if want_gdn != isinstance(blk.mixer, GatedDeltaNet):   # swap the operator
                    mixer = (GatedDeltaNet(width, heads, mult, std, out_std) if want_gdn
                             else Attention(width, heads, mult, std, out_std))
                    self.blocks[i] = Block(mixer, width, mult, std, out_std)
                    blk = self.blocks[i]
            if isinstance(blk.mixer, GatedDeltaNet):
                blk.mixer.__class__ = _gdn_class('both' if is_top else delta)
            else:
                blk.mixer.__class__ = (Attention if is_top or attn == 'bidir'
                                       else CausalAttention)

    def layer_kinds(self):
        """Per block: ``'gdn:<dirs>'`` or ``'attn:<bidir|causal>'``."""
        out = []
        for blk in self.blocks:
            m = blk.mixer
            if isinstance(m, GatedDeltaNet):
                out.append('gdn:' + {GatedDeltaNet: 'both', CausalGatedDeltaNet: 'l2r',
                                      ReverseGatedDeltaNet: 'r2l'}[type(m)])
            else:
                out.append('attn:' + ('causal' if type(m) is CausalAttention else 'bidir'))
        return out


# --- registered variants: one model name per fixed setting of the fields ---
VARIANTS = {
    # A (xs): which operator needs to be acausal, and which direction
    'hybrid4_mix_gdnl2r': dict(delta='l2r', attn='bidir', top=0, top_op='native'),
    'hybrid4_mix_attncausal': dict(delta='both', attn='causal', top=0, top_op='native'),
    'hybrid4_mix_gdnr2l': dict(delta='r2l', attn='bidir', top=0, top_op='native'),
    # B (s): causal trunk, last k layers bidirectional
    'hybrid4_causal_top1': dict(delta='l2r', attn='causal', top=1, top_op='native'),
    'hybrid4_causal_top2': dict(delta='l2r', attn='causal', top=2, top_op='native'),
    'hybrid4_causal_top3': dict(delta='l2r', attn='causal', top=3, top_op='native'),
    'hybrid4_causal_top1gdn': dict(delta='l2r', attn='causal', top=1, top_op='gdn'),
}


def _variant(name, arch):
    mod = types.ModuleType(f'{__name__}.{name}',
                           f'{name}: hybrid4_mix with {arch} ({__name__}).')
    mod.SIZES, mod.LR, mod.SCALING_RULE, mod.ARCH = SIZES, LR, SCALING_RULE, dict(arch)
    mod.build = lambda size, spec, _a=dict(arch): Hybrid4Mix(spec, **SIZES[size], **_a)
    mod.__file__ = __file__
    sys.modules[mod.__name__] = mod
    register(name)(mod)


for _name, _arch in VARIANTS.items():
    _variant(_name, _arch)
