"""hybrid4c: hybrid4 (``mapping/models/se2d/hybrid4.py``) ported to the
continuous SE(2) task (``CONTRACT_SE2C.md``).

What changes, and nothing else:

1. **Tokeniser for a 15 x 15 view.** Each observation is cut into 25
   patches of 3 x 3 (the same patch size as hybrid4, so 25 tokens per view
   instead of 9), through the same ``Linear(9, w)``, plus one of 25
   learned patch-position vectors shared by every step. Each query has 25
   tokens, one per output 3 x 3 patch, with 25 learned query-position
   vectors, and each is read out to its 9 pixels. Sequence: 33 x 26 - 1 +
   8 x 25 = 1057 tokens (hybrid4: 401).
2. **Odometry tokens from the shared encoding.** The raw relative pose
   (dy, dx, dtheta) goes through ``mapping.se2c.encoding.encode_odometry``
   (scaled scalars and octave Fourier features of the translation, cos
   and sin of m dtheta for m = 1..4; 50 features) and one ``Linear(50,
   w)``. This replaces hybrid4's ``Embedding(16, w)`` over the discrete
   symbol: a table lookup becomes a linear map of smooth features.
3. **Query tokens from the query encoding.** ``encode_query`` of the raw
   query pose through one ``Linear(50, w)``, added to each of the query's
   25 tokens. This replaces hybrid4's offset Fourier features plus the
   4-way turn embedding. The Linear has PyTorch's default init, like
   hybrid4's ``offset_embed``.

Everything else is hybrid4's, imported from its module so it cannot
drift: the bidirectional Gated DeltaNet with beta in (0, 2) and the
width-4 short convolution, softmax attention with RoPE on the step index,
SwiGLU MLPs, pre-norm blocks, alternation (delta rule first), the
width ladder, the muP multipliers and initialisation, and placement:
query q's 25 tokens sit right after observation ``query_time[q]`` and
before odometry token ``query_time[q]``, with RoPE step
``query_time[q]``; with every query at the last step this is
"queries at the end".

Mechanism trace: how the turn angle reaches the delta rule
----------------------------------------------------------
In the discrete model the trained right-to-left delta-rule head gave
every odometry token beta = 2 (a reflection ``I - 2 k k^T``) and rotated
one 2-D state subspace by exactly the turn; two reflections whose
normals are at angle phi compose into a rotation by 2 phi, so a turn by
theta needs a pair of keys at relative angle theta / 2 in that plane.
With four turns, k could be a lookup per symbol. Here theta is
continuous, and the path is:

1. ``encode_odometry`` puts [cos m theta, sin m theta], m = 1..4, into
   the odometry token, through ``odo_embed`` (linear), so the residual
   stream at odometry token t is an affine function of these features
   plus the translation features.
2. In the first Gated DeltaNet layer, after the block's LayerNorm (a
   per-token rescaling), q, k and v are linear projections of the token;
   the short convolution then mixes, per channel, the projections of the
   token and the three before it in the head's own scan order, followed
   by SiLU and L2 normalisation of k. In the right-to-left head the scan
   meets odometry token t right before the patches of observation t
   (reversed: 24, 23, ...), or before the tokens of a query asked at t
   when there is one; the convolution windows of the next three tokens
   all see odometry token t. One turn can therefore set up to four keys,
   which is where a second, angle-dependent reflection can come from.
   (Left to right, odometry token t precedes patches 0, 1, 2 of
   observation t+1.)
3. beta = 2 sigmoid(w . LN(x) + b) is computed per token WITHOUT the
   convolution, so the odometry token's own beta is a smooth function of
   its features; beta = 2 on every odometry token is reachable and
   angle-independent, as in the discrete model (a zero turn is two
   reflections with the same key).
4. What has to be learned is the chart theta -> k(theta / 2). A smooth
   route exists for most angles: a key proportional to (1 + cos theta,
   sin theta) = 2 cos(theta/2) (cos(theta/2), sin(theta/2)) is affine in
   the m = 1 features and L2 normalisation turns it into exactly the
   half-angle direction (SiLU is near-linear for positive pre-activations).
   It fails at theta = +-pi, where the vector vanishes. That is not a
   detail of this route: a reflection depends on k only up to sign, and
   a key that is a continuous function of the turn on the whole circle
   cannot give a half-angle reflection for every theta (the half-angle
   line winds once around the projective line while theta winds once;
   no continuous unit vector does that). So a continuous model must put
   a cut somewhere in theta, where k swings through out of the plane or
   beta drops below 2 (a contraction, |lambda| < 1, rather than a
   rotation). Prediction for the probes: on held-out turns the
   eigenangle tracks theta (slope 1) except in a band around one angle,
   probably near +-pi, where the error per step is larger and the
   transition contracts.

Audit (hybrid4's items hold with the following replaced or added):
10. Odometry: the raw float pose, encoded by the fixed shared feature map
    and a learned Linear; never rotated, composed or accumulated outside
    learned layers.
11. Query pose: the raw float pose, encoded by the same fixed feature map
    and a learned Linear, added to the query tokens; never combined with
    the odometry by arithmetic.
12. Observations are patched as given, facing their own headings.
13. Patch and query positions are indices 0..24 within a view, a sequence
    index; step enters only through RoPE and sequence order, and query
    placement uses only ``query_time``.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapping.models.se2d.hybrid4 import (BASE_WIDTH, HEAD_DIM, LR,  # noqa: F401
                                        SCALING_RULE, SIZES, Attention, Block,
                                        GatedDeltaNet, MuLinear, rope_cos_sin)
from mapping.models.query_mode import QueryLayout, check_query_mode
from mapping.models.scan_precision import check_scan_precision, set_scan_precision
from mapping.models.se2c import register
from mapping.se2c.encoding import (ODOMETRY_FEATURES, QUERY_FEATURES,
                                   encode_odometry, encode_query)

PATCH = 3


def patchify(x, p=PATCH):
    """``(..., S, S)`` -> ``(..., (S/p)^2, p*p)``, row-major patches."""
    *lead, s, _ = x.shape
    n = s // p
    x = x.reshape(*lead, n, p, n, p).transpose(-3, -2)
    return x.reshape(*lead, n * n, p * p)


def unpatchify(x, s, p=PATCH):
    """Inverse of ``patchify``."""
    *lead, _, _ = x.shape
    n = s // p
    x = x.reshape(*lead, n, n, p, p).transpose(-3, -2)
    return x.reshape(*lead, s, s)


def token_order(query_time, t1, q, P):
    """Permutation ``(b, t1*(P+1) - 1 + q*P)`` of the layout [obs_0 x P,
    odo_0, ..., obs_T x P, query tokens] that places query q's P tokens
    after observation ``query_time[q]``: a stable sort by (step, kind)
    with kinds observation < query < odometry."""
    dev = query_time.device
    b = query_time.shape[0]
    steps = torch.arange(t1, device=dev)
    kind_obs = (steps[:, None] * 3).expand(t1, P)
    kind_odo = (steps * 3 + 2)[:, None]
    ctx = torch.cat([kind_obs, kind_odo], 1).reshape(-1)[:-1]       # (L_ctx,)
    qry = (query_time[:, :, None] * 3 + 1).expand(-1, -1, P).reshape(b, -1)
    key = torch.cat([ctx[None].expand(b, -1), qry], 1)
    n = key.shape[1]
    key = key * n + torch.arange(n, device=dev)        # ties: layout order
    return torch.argsort(key, 1)


class Hybrid4c(nn.Module):
    QUERY_MODES = ('mixed', 'inert')    # mapping/models/query_mode.py
    SCAN_PRECISIONS = ('tf32', 'ieee')  # mapping/models/scan_precision.py

    def __init__(self, spec, width, mixer='hybrid', patch=PATCH, query_mode='mixed',
                 scan_precision='tf32'):
        super().__init__()
        assert spec.view % patch == 0, 'view must be a multiple of the patch'
        check_query_mode(self, query_mode)
        check_scan_precision(self, scan_precision)
        self.query_mode = query_mode
        self.scan_precision = scan_precision
        self.spec = spec
        self.width = width
        self.patch = patch       # 3 for every rung; 5 only for the cost check
        self.P = (spec.view // patch) ** 2
        heads = width // HEAD_DIM
        mixers = width // HEAD_DIM
        mult = BASE_WIDTH / width
        std = 0.5 / math.sqrt(width)
        out_std = std / math.sqrt(2 * mixers)
        self.patch_embed = nn.Linear(patch * patch, width)
        self.patch_pos = nn.Embedding(self.P, width)
        self.query_pos = nn.Embedding(self.P, width)
        self.odo_embed = nn.Linear(ODOMETRY_FEATURES, width)      # SE2C
        self.query_embed = nn.Linear(QUERY_FEATURES, width)       # SE2C
        self.blocks = nn.ModuleList()
        for i in range(mixers):
            if mixer == 'hybrid' and i % 2 == 0:
                m = GatedDeltaNet(width, heads, mult, std, out_std)
            else:
                m = self.attention(width, heads, mult, std, out_std)
            self.blocks.append(self.block(m, width, mult, std, out_std))
        self.norm = nn.LayerNorm(width)
        self.readout = MuLinear(width, patch * patch, mult, std, zero=True)
        for emb in (self.patch_pos, self.query_pos):
            nn.init.normal_(emb.weight, std=0.02)
        set_scan_precision(self, scan_precision, GatedDeltaNet)

    # the mixer and block classes (vanilla4c swaps in vanilla4's)
    attention = Attention
    block = Block

    def forward(self, obs, odometry, query_pose, query_time=None):
        b, t1 = obs.shape[:2]
        q = query_pose.shape[1]
        w, P, S = self.width, self.P, self.spec.view
        dev = obs.device
        if query_time is None:
            query_time = torch.full((b, q), t1 - 1, device=dev,
                                    dtype=torch.long)
        idx = torch.arange(P, device=dev)
        steps = torch.arange(t1, device=dev)
        obs_tok = (self.patch_embed(patchify(obs, self.patch))
                   + self.patch_pos(idx))                              # (b,t1,P,w)
        odo = self.odo_embed(encode_odometry(odometry[..., :2],
                                             odometry[..., 2]))       # (b,T,w)
        odo_tok = F.pad(odo, (0, 0, 0, 1))                             # (b,t1,w)
        seq = torch.cat([obs_tok, odo_tok[:, :, None]], 2)
        seq = seq.reshape(b, t1 * (P + 1), w)[:, :-1]
        qry = (self.query_embed(encode_query(query_pose[..., :2],
                                             query_pose[..., 2]))[:, :, None]
               + self.query_pos(idx))                                  # (b,q,P,w)
        x = torch.cat([seq, qry.reshape(b, q * P, w)], 1)
        ctx_pos = steps[:, None].expand(t1, P + 1).reshape(-1)[:-1]
        pos = torch.cat([ctx_pos[None].expand(b, -1),
                         query_time[:, :, None].expand(-1, -1, P).reshape(b, -1)],
                        1)
        order = token_order(query_time, t1, q, P)
        x = torch.gather(x, 1, order[..., None].expand(-1, -1, w))
        pos = torch.gather(pos, 1, order)
        cos, sin = rope_cos_sin(pos, HEAD_DIM)
        if self.query_mode == 'inert':
            layout = QueryLayout(order, t1 * (P + 1) - 1, q, P)
            for blk in self.blocks:
                x = blk(x, cos, sin, layout)
        else:
            for blk in self.blocks:
                x = blk(x, cos, sin)
        inverse = torch.argsort(order, 1)
        x = torch.gather(x, 1, inverse[..., None].expand(-1, -1, w))
        y = self.readout(self.norm(x[:, x.shape[1] - q * P:]))        # (b,q*P,9)
        return unpatchify(y.view(b, q, P, self.patch ** 2), S, self.patch)


def build(size, spec):
    return Hybrid4c(spec, **SIZES[size])


register('hybrid4c')(sys.modules[__name__])
