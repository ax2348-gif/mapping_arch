"""Recurrent baseline: a residual bidirectional LSTM encoder with a
state-bridged LSTM decoder for the queries.

This is the classic sequence-to-sequence recipe (encoder RNN, decoder RNN
seeded from the encoder's final state) brought up to current practice:
pre-norm residual blocks around each cuDNN LSTM layer, orthogonal recurrent
init with the forget gate biased open, a bidirectional encoder since the
setting is offline, and a Fourier-feature embedding for the integer query
offsets. Everything the model knows about the episode has to pass through
the encoder's final (h, c) states: there is no attention from the queries
back over the sequence, so the decoder reads from a fixed-size state and
nothing else. That is what makes it the recurrent reference point on the
scaling plot.

Tokenisation. The episode becomes 2T + 1 = 65 tokens, interleaved
    obs_0, dir_0, obs_1, dir_1, ..., dir_{T-1}, obs_T
so every input event is its own token: observation tokens are a two-layer
MLP of the 81 raw pixels, direction tokens are a 4-entry embedding table,
and a 2-entry token-type embedding is added. The recurrence sees them in
order and nothing else; no sequence-index encoding is added because an RNN
carries order in its state.

Encoder. ``depth`` residual blocks, each ``x + LSTM(LayerNorm(x))``, with a
bidirectional LSTM of ``width // 2`` units per direction so the block
output is ``width`` and the residual stream is a plain sum. The backward
direction is the point of going bidirectional: the queries are expressed
relative to the last observation, so a recurrence that starts at the end
of the episode accumulates exactly the quantities a query is asked in.

Decoder (readout = 'recurrent', the default and the pure variant). Each
query is embedded once and fed to a unidirectional residual LSTM stack of
the same depth and width, whose layer-l initial (h, c) is a linear bridge
from encoder layer l's final (h, c). The query token is presented for
``think`` steps so the decoder can take a few recurrent updates before the
last step's output is normalised and decoded to 81 pixels by a small MLP.
Queries are independent, so the decoder runs them as a batch of B*Q
sequences, each started from a copy of the same encoder state.

Variants kept for ablation, not on the ladder:
    readout='mlp'   pure: an MLP on [encoder final states, query embedding]
    readout='attn'  HYBRID, not recurrent: the query embedding attends over
                    the encoder output sequence with one multi-head
                    attention layer before the MLP head. Label any run of
                    this as "lstm (hybrid attention readout)".

Training notes. cuDNN LSTM is used for every recurrent layer (one kernel
launch per layer and direction, 65 steps each), so the sequential scan is
not the bottleneck. The recurrent layers run in fp32 (TF32) even under the
harness's bf16 autocast: cuDNN RNN has no bf16 kernel, autocast silently
runs it in fp16, and without loss scaling the per-pixel MSE gradients
(~1e-5) sit in fp16's subnormal range; in a 2000-step xs check the fp16
path lost to fp32 (eval mse 0.0952 vs 0.0939). Weights: Xavier-uniform input matrices, orthogonal
recurrent matrices per gate, forget-gate bias 1, other biases 0. The
harness clips the global gradient norm at 1.0, which is what a residual
LSTM stack wants; no extra clipping is done inside the model.

Audit: where positional or geometric information enters the network.
  1. ``directions`` -> ``dir_in`` (nn.Embedding(4, width)), one token per
     step, placed in sequence order only. Nothing is summed, cumulated or
     turned into coordinates outside the learned LSTM layers.
  2. ``query_offset`` -> ``OffsetEmbedding``: the raw (dy, dx) integers,
     scaled by offset_max and expanded in fixed sin/cos features of the
     scalars themselves, then an MLP. The offsets are used only as a raw
     input to the decoder; they are never combined with directions,
     observations, or anything else by hand-written arithmetic, and
     they never index anything.
  3. Token order is the only sequence-index information, and it enters
     implicitly through the recurrence; there is no positional encoding
     of any kind, so nothing is keyed on world geometry.
  4. The 'attn' variant attends over the encoder output with no positional
     bias at all (keys carry order only through the recurrence).
  5. ``obs`` enters as flat pixels per token. No canvas, no scatter or
     gather by coordinates, no absolute positions anywhere in the module.
"""
import math
import sys

import torch
import torch.nn as nn

from mapping.models import register

# width x sqrt(2) per rung (rounded to a multiple of 64); depth +1 every
# second rung. Params ~ 14 * width^2 * depth, see SCALING_RULE.
from mapping.models import ladder  # noqa: E402
SIZES = ladder('lstm')   # conf/model/size/lstm/*.yaml

SCALING_RULE = (
    'One-parameter family indexed by rung r = 0..6: width = 128 * 2^(r/2) '
    'rounded to a multiple of 64 (128, 192, 256, 384, 512, 768, 1024) and '
    'depth = 2 + r // 2 (2, 2, 3, 3, 4, 4, 5); the decoder has the same '
    'depth and width as the encoder, the bidirectional encoder uses '
    'width/2 units per direction, and the decoder "think" steps (4), the '
    'Fourier feature count (16 per axis) and the head width are constant. '
    'Parameters go as ~14 * width^2 * depth, about x160 across the ladder. '
    'Learning rate: constant across rungs at the harness default '
    '(3e-4, warmup 100, cosine, AdamW betas 0.9/0.95, wd 0.1, global clip '
    '1.0). This is in the usual stable range for residual LSTM stacks at '
    'every width here; orthogonal recurrent init and forget bias 1 are set '
    'inside the model so no per-rung tuning is needed.'
)


def _init_lstm(rnn):
    """Xavier input weights, orthogonal recurrent weights (per gate),
    forget-gate bias 1, everything else 0. Gate order in torch: i f g o."""
    h = rnn.hidden_size
    with torch.no_grad():
        for name, p in rnn.named_parameters():
            if name.startswith('weight_ih'):
                for k in range(4):
                    nn.init.xavier_uniform_(p[k * h:(k + 1) * h])
            elif name.startswith('weight_hh'):
                for k in range(4):
                    nn.init.orthogonal_(p[k * h:(k + 1) * h])
            elif name.startswith('bias_ih'):
                p.zero_()
                p[h:2 * h].fill_(1.0)
            elif name.startswith('bias_hh'):
                p.zero_()


class ResLSTM(nn.Module):
    """Pre-norm residual block around one cuDNN LSTM layer."""

    def __init__(self, width, bidirectional):
        super().__init__()
        if bidirectional and width % 2:
            raise ValueError('bidirectional needs an even width')
        hidden = width // 2 if bidirectional else width
        self.norm = nn.LayerNorm(width)
        self.rnn = nn.LSTM(width, hidden, batch_first=True,
                           bidirectional=bidirectional)
        _init_lstm(self.rnn)

    def forward(self, x, state=None):
        xn = self.norm(x)
        # cuDNN RNN has no bf16 path, so autocast would run it in fp16 with
        # no loss scaling; per-pixel MSE gradients (~1e-5) are subnormal
        # there. Run the recurrence in fp32 (TF32 matmuls); the rest of the
        # network stays under the harness's bf16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            if state is not None:
                state = (state[0].float(), state[1].float())
            y, (h, c) = self.rnn(xn.float(), state)
        return x + y, (h, c)


class OffsetEmbedding(nn.Module):
    """Raw (dy, dx) integer offsets -> width. Scaled scalars plus fixed
    sin/cos features of those scalars at log-spaced periods from 3 px to
    4 * offset_max, then a two-layer MLP. Nothing but the offset goes in."""

    def __init__(self, width, offset_max, n_freq=16):
        super().__init__()
        self.offset_max = float(offset_max)
        periods = torch.logspace(math.log10(3.0),
                                 math.log10(4.0 * offset_max), n_freq)
        self.register_buffer('freq', 2 * math.pi / periods)
        self.mlp = nn.Sequential(nn.Linear(2 + 4 * n_freq, width),
                                 nn.GELU(), nn.Linear(width, width))

    def forward(self, offset):                        # (B, Q, 2) long
        x = offset.float()
        ang = x[..., None] * self.freq                # (B, Q, 2, F)
        feats = torch.cat([x / self.offset_max, ang.sin().flatten(-2),
                           ang.cos().flatten(-2)], -1)
        return self.mlp(feats)


class RecurrentMapper(nn.Module):
    def __init__(self, spec, width, depth, bidirectional=True, think=4,
                 readout='recurrent', n_freq=16):
        super().__init__()
        if readout not in ('recurrent', 'mlp', 'attn'):
            raise ValueError(readout)
        s, t = spec.view, spec.steps
        self.spec, self.width, self.depth = spec, width, depth
        self.think, self.readout = think, readout
        self.n_tokens = 2 * t + 1

        # tokenisation
        self.obs_in = nn.Sequential(nn.Linear(s * s, width), nn.GELU(),
                                    nn.Linear(width, width))
        self.dir_in = nn.Embedding(spec.directions, width)
        self.type_in = nn.Embedding(2, width)
        self.offset_in = OffsetEmbedding(width, spec.offset_max, n_freq)

        # encoder
        self.encoder = nn.ModuleList(
            [ResLSTM(width, bidirectional) for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(width)

        # readout
        if readout == 'recurrent':
            self.bridge_h = nn.ModuleList(
                [nn.Linear(width, width) for _ in range(depth)])
            self.bridge_c = nn.ModuleList(
                [nn.Linear(width, width) for _ in range(depth)])
            self.decoder = nn.ModuleList(
                [ResLSTM(width, bidirectional=False) for _ in range(depth)])
            head_in = width
        elif readout == 'mlp':
            head_in = 2 * depth * width + width
        else:  # 'attn': HYBRID
            self.attn = nn.MultiheadAttention(width, max(1, width // 64),
                                              batch_first=True)
            self.attn_norm = nn.LayerNorm(width)
            head_in = width
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(nn.Linear(head_in, width), nn.GELU(),
                                  nn.Linear(width, s * s))

    def tokens(self, obs, directions):
        b = obs.shape[0]
        o = self.obs_in(obs.flatten(2)) + self.type_in.weight[0]
        d = self.dir_in(directions) + self.type_in.weight[1]
        x = o.new_zeros(b, self.n_tokens, self.width)
        x[:, 0::2] = o
        x[:, 1::2] = d
        return x

    def forward(self, obs, directions, query_offset):
        b, q = query_offset.shape[:2]
        s, w = self.spec.view, self.width

        x = self.tokens(obs, directions)
        states = []
        for block in self.encoder:
            x, (h, c) = block(x)
            # (dirs, B, hidden) -> (B, width): forward and backward halves
            states.append((h.transpose(0, 1).reshape(b, w),
                           c.transpose(0, 1).reshape(b, w)))
        qe = self.offset_in(query_offset)                      # (B, Q, W)

        if self.readout == 'recurrent':
            y = qe.reshape(b * q, 1, w).expand(-1, self.think, -1)
            for block, (h, c), lin_h, lin_c in zip(self.decoder, states,
                                                   self.bridge_h,
                                                   self.bridge_c):
                h0 = lin_h(h)[:, None].expand(-1, q, -1).reshape(1, b * q, w)
                c0 = lin_c(c)[:, None].expand(-1, q, -1).reshape(1, b * q, w)
                y, _ = block(y, (h0.contiguous(), c0.contiguous()))
            z = y[:, -1]                                        # (B*Q, W)
        elif self.readout == 'mlp':
            flat = torch.cat([t for st in states for t in st], -1)  # (B, 2LW)
            z = torch.cat([flat[:, None].expand(-1, q, -1), qe], -1)
        else:  # 'attn': HYBRID readout
            mem = self.enc_norm(x)
            a, _ = self.attn(self.attn_norm(qe), mem, mem, need_weights=False)
            z = qe + a
        out = self.head(self.head_norm(z))
        return out.reshape(b, q, s, s)


def build(size, spec):
    return RecurrentMapper(spec, **SIZES[size])


register('lstm')(sys.modules[__name__])
