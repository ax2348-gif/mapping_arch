"""Blind-probe models: a minimal hybrid4-style stack and its controls.

Every model is width 64 (one head of dim 32 per scan direction), two
token-mixer blocks, each followed by a SwiGLU MLP, pre-norm, with
hybrid4's muP multipliers and initialisation. The attention, MLP,
MuLinear and RoPE code is imported from ``mapping/models/se2/hybrid4.py``
unchanged; the complex-diagonal mixer from ``hybrid4_cdiag.py``.

  gdn     [Gated DeltaNet, beta in (0, 2), short conv 4] -> [attention]
  gdn_b1  the same with beta = sigmoid(.) in (0, 1)
  attn    [attention] -> [attention]                     (vanilla4-style)
  cdiag   [complex-diagonal recurrence] -> [attention]   (hybrid4_cdiag's mixer)

Sequence (2T + 1 + Q tokens), sorted by step:

  anchor_0  odo_1 anchor_1 [queries at 1]  ...  odo_T anchor_T [queries at T]

* odo_i: the encoded odometry of step i (pose i in the frame of pose i-1)
  through one linear map.
* anchor_i: one learned vector, the same at every step. It stands where
  hybrid4 had the observation patches: a token per pose that carries no
  input, so with the short conv it can supply the second reflection of
  step i (two Householder updates per step).
* query (s, t): placed right after anchor_t, at RoPE position t; a learned
  vector plus Fourier features of the lag (t - s) / T through one linear
  map, as the mapping task's query tokens carry their offset.
RoPE position of odo_i and anchor_i is i.

Readout: final LayerNorm and one zero-initialised linear map from each
query token to (dx, dy, cos, sin), the mapping task's readout with 4
outputs instead of 9 pixels. No pose head beyond that linear map.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

sys.path.insert(0, '.')
from mapping.models.se2 import hybrid4 as h4                       # noqa: E402
from mapping.models.se2.hybrid4_cdiag import ComplexDiagonal        # noqa: E402

HEAD_DIM, DECAY_TAU, BASE_WIDTH = h4.HEAD_DIM, h4.DECAY_TAU, h4.BASE_WIDTH
KINDS = ('gdn', 'gdn_b1', 'attn', 'cdiag')
LAG_FREQS = 6


class GatedDeltaNet(nn.Module):
    """hybrid4's GatedDeltaNet with the beta range as a parameter
    (``beta_max`` 2: hybrid4; 1: hybrid4_beta1), and the per-token gate
    computation split out (``gates``) so the probe reads exactly what the
    scan uses. Parameter names match hybrid4's layer; ``check_against_h4``
    asserts identical outputs with shared weights."""

    def __init__(self, width, heads, mult, std, out_std, conv=4, beta_max=2.0):
        super().__init__()
        assert heads % 2 == 0
        self.heads, self.d, self.conv, self.beta_max = heads, HEAD_DIM, conv, beta_max
        inner = heads * HEAD_DIM
        self.q = h4.MuLinear(width, inner, mult, std)
        self.k = h4.MuLinear(width, inner, mult, std)
        self.v = h4.MuLinear(width, inner, mult, std)
        self.short = nn.Conv1d(3 * inner, 3 * inner, conv, groups=3 * inner, bias=False)
        self.gate = h4.MuLinear(width, inner, mult, std)
        self.decay = h4.MuLinear(width, heads, mult, std)
        nn.init.constant_(self.decay.bias, 3.0)
        self.beta = h4.MuLinear(width, heads, mult, std)
        nn.init.constant_(self.beta.bias, 0.0)   # beta = beta_max / 2 at init, as hybrid4 / hybrid4_beta1
        self.norm = nn.RMSNorm(HEAD_DIM)
        self.out = h4.MuLinear(inner, width, mult, out_std)

    def gates(self, x):
        """q, k, v (b, l, h, d), log decay g and beta (b, l, h), all in each
        head's own scan order (right-to-left heads reversed)."""
        b, l, _ = x.shape
        h, d, half = self.heads, self.d, self.heads // 2
        qkv = torch.stack([self.q(x), self.k(x), self.v(x)], 2).view(b, l, 3, h, d)
        g = F.logsigmoid(self.decay(x).float()) / DECAY_TAU
        beta = self.beta_max * torch.sigmoid(self.beta(x).float())
        seq = torch.cat([qkv[:, :, :, :half], qkv[:, :, :, half:].flip(1)], 3)
        seq = F.pad(seq.reshape(b, l, 3 * h * d).transpose(1, 2), (self.conv - 1, 0))
        seq = F.silu(self.short(seq)).transpose(1, 2).view(b, l, 3, h, d)
        q, k, v = seq.unbind(2)
        q = F.normalize(q.float(), dim=-1).to(v.dtype) * d ** -0.5
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        gh = torch.cat([g[:, :, :half], g[:, :, half:].flip(1)], 2)
        bh = torch.cat([beta[:, :, :half], beta[:, :, half:].flip(1)], 2)
        return q, k, v, gh, bh

    def forward(self, x):
        b, l, _ = x.shape
        h, d, half = self.heads, self.d, self.heads // 2
        q, k, v, g, beta = self.gates(x)
        o, _ = chunk_gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(),
                                      g.contiguous(), beta.contiguous(), scale=1.0)
        o = torch.cat([o[:, :, :half], o[:, :, half:].flip(1)], 2)
        o = self.norm(o).to(x.dtype) * F.silu(self.gate(x)).view(b, l, h, d)
        return self.out(o.reshape(b, l, h * d))


def lag_features(lag, T):
    o = lag.float()[..., None] / T
    f = math.pi * 2.0 ** torch.arange(LAG_FREQS, device=lag.device).float()
    return torch.cat([o, (o * f).sin(), (o * f).cos()], -1)


class BlindProbe(nn.Module):
    def __init__(self, kind, feat_dim, T=32, width=64):
        super().__init__()
        assert kind in KINDS
        self.kind, self.T, self.width = kind, T, width
        heads = width // HEAD_DIM
        mixers = 2
        mult = BASE_WIDTH / width
        std = 0.5 / math.sqrt(width)
        out_std = std / math.sqrt(2 * mixers)
        if kind == 'gdn':
            first = GatedDeltaNet(width, heads, mult, std, out_std, beta_max=2.0)
        elif kind == 'gdn_b1':
            first = GatedDeltaNet(width, heads, mult, std, out_std, beta_max=1.0)
        elif kind == 'cdiag':
            first = ComplexDiagonal(width, heads, mult, std, out_std)
        else:
            first = h4.Attention(width, heads, mult, std, out_std)
        second = h4.Attention(width, heads, mult, std, out_std)
        self.blocks = nn.ModuleList([h4.Block(m, width, mult, std, out_std)
                                     for m in (first, second)])
        self.odo_embed = nn.Linear(feat_dim, width)
        self.lag_embed = nn.Linear(1 + 2 * LAG_FREQS, width)
        self.anchor = nn.Parameter(torch.randn(width) * 0.02)
        self.query = nn.Parameter(torch.randn(width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.readout = h4.MuLinear(width, 4, mult, std, zero=True)

    @staticmethod
    def layout(t, T):
        """Sort order of [anchor_0..anchor_T, odo_1..odo_T, queries] into
        sequence order, the RoPE position of every token (layout order),
        and token labels (layout order): kind 0 anchor, 1 odo, 2 query,
        with the step."""
        dev = t.device
        b, Q = t.shape
        steps = torch.arange(T + 1, device=dev)
        key = torch.cat([(3 * steps + 1)[None].expand(b, -1),          # anchors
                         (3 * steps[1:])[None].expand(b, -1),          # odometry
                         3 * t + 2], 1).double()                       # queries
        key = key + torch.arange(key.shape[1], device=dev, dtype=key.dtype) * 1e-4
        order = torch.argsort(key, 1)
        pos = torch.cat([steps[None].expand(b, -1), steps[1:][None].expand(b, -1), t], 1)
        kind = torch.cat([torch.zeros(T + 1), torch.ones(T), 2 * torch.ones(Q)]).long().to(dev)
        step = torch.cat([steps[None].expand(b, -1), steps[1:][None].expand(b, -1), t], 1)
        return order, pos, kind[None].expand(b, -1), step

    def embed(self, feats, s, t):
        b, T, _ = feats.shape
        Q = t.shape[1]
        anc = self.anchor.expand(b, T + 1, -1)
        odo = self.odo_embed(feats)
        qry = self.query + self.lag_embed(lag_features(t - s, self.T))
        x = torch.cat([anc, odo, qry], 1)
        order, pos, _, _ = self.layout(t, T)
        x = torch.gather(x, 1, order[..., None].expand(-1, -1, self.width))
        pos = torch.gather(pos, 1, order)
        return x, pos, order

    def forward(self, feats, s, t, stop_at=None):
        """feats (b, T, F) encoded odometry, s, t (b, Q). Returns (b, Q, 4).
        ``stop_at=i`` returns the input of block i's mixer (sequence order)."""
        x, pos, order = self.embed(feats, s, t)
        cos, sin = h4.rope_cos_sin(pos, HEAD_DIM)
        for i, block in enumerate(self.blocks):
            if stop_at == i:
                return block.norm1(x), order
            x = block(x, cos, sin)
        x = torch.gather(x, 1, torch.argsort(order, 1)[..., None].expand(-1, -1, self.width))
        Q = t.shape[1]
        return self.readout(self.norm(x[:, -Q:]))


def check_against_h4(device='cuda'):
    """Our GatedDeltaNet (beta_max 2) equals hybrid4's layer with the same
    weights; beta_max 1 gives beta = sigmoid; layout is a valid order."""
    torch.manual_seed(0)
    w, heads = 64, 2
    mult, std = BASE_WIDTH / w, 0.5 / math.sqrt(w)
    ours = GatedDeltaNet(w, heads, mult, std, std).to(device)
    ref = h4.GatedDeltaNet(w, heads, mult, std, std).to(device)
    ref.load_state_dict(ours.state_dict())
    with torch.no_grad():
        nn.init.normal_(ours.beta.bias, std=1.0); ref.beta.bias.copy_(ours.beta.bias)
        x = torch.randn(3, 81, w, device=device)
        a, b = ours(x), ref(x)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    ours1 = GatedDeltaNet(w, heads, mult, std, std, beta_max=1.0).to(device)
    _, _, _, _, beta = ours1.gates(x)
    assert beta.max() < 1.0
    t = torch.tensor([[1, 5, 5, 32]], device=device)
    order, pos, kind, step = BlindProbe.layout(t, 32)
    seq_kind, seq_step = kind[0][order[0]].tolist(), step[0][order[0]].tolist()
    assert seq_kind[:3] == [0, 1, 0] and seq_step[:3] == [0, 1, 1]
    for i in range(1, len(seq_kind)):                       # steps non-decreasing,
        assert seq_step[i] >= seq_step[i - 1]               # odo < anchor < query in a step
        if seq_step[i] == seq_step[i - 1]:
            rank = {1: 0, 0: 1, 2: 2}
            assert rank[seq_kind[i]] >= rank[seq_kind[i - 1]]
    print('model checks passed: GatedDeltaNet == hybrid4 layer; layout valid')


if __name__ == '__main__':
    check_against_h4()
