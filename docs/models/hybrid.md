# hybrid: attention interleaved with gated linear attention

**File** `mapping/models/hybrid.py` (audit section in the docstring).

## Architecture

A pre-norm transformer in which every other token mixer is a gated linear
attention (GLA) layer, as in the hybrid SSM/attention stacks of the LLM
literature (Jamba, Samba, Zamba, Griffin). Blocks are `[GLA, MLP, Attn, MLP]`
repeated; LayerNorm pre-norm, GELU MLP at ratio 4, GPT-2 init with residual
outputs scaled by `1/sqrt(2*mixers)`, zero-initialised readout.

**Tokens (401).** ViT-style 3x3 patches, nine per observation, with learned
patch-index (0..8) and time-step (0..32) embeddings; one direction token per
step (`nn.Embedding(4, w)` + time-step embedding) interleaved in sequence
order between the two observation groups it connects; nine query tokens per
query (one per output patch, like MAE decoder mask tokens) carrying the
patch-index embedding plus an embedding of the raw offset (Fourier features
at 8 octave-spaced frequencies per axis plus the scaled scalars, one linear
layer), appended at the end. Each query token decodes to its 3x3 patch.
Interleaving suits a scan: it passes each direction token once, between the
content it separates, so one running state suffices; a separate direction
segment would make the scan hold 32 partial sums.

**Recurrence.** GLA with a scalar data-dependent decay per head (Mamba-2 /
gated-retention form): `h_t = a_t h_{t-1} + k_t v_t^T`, `o_t = q_t h_t`,
`a_t = sigmoid(z_t)^(1/16)` initialised near 1. Pure-PyTorch chunked
parallel scan in float32 (chunk 32: masked intra-chunk matmul plus a 13-step
sequential carry), checked against a sequential reference. **Bidirectional
in the Vision-Mamba sense:** half the heads scan left-to-right, half
right-to-left, no shared state. Scan output is per-head RMS-normalised,
multiplied by a SiLU output gate, and projected.

## Inductive bias over the vanilla transformer

The direction stream has to become a running sum before any gather can be
keyed on it. Softmax attention can only approximate a prefix sum with weight
spread over O(T) tokens and a readout that counts; a linear recurrence with
`a_t -> 1` *is* a prefix sum of the `k v` outer products, in one layer with
constant parameters, for every prefix at once. With that summary in every
token's residual stream, the softmax layers only do content matching and
copying. The added bias is exactly a cumulative-aggregation primitive along
the sequence; nothing is keyed on coordinates and no sum is written by hand.

## Scaling rule

One-parameter family in the width `w`: heads `w/32` (head dim 32, half per
scan direction), per-head state 32x32 (total `32w`), mixers `w/32` (`w/64`
GLA+attention pairs). `LR[size] = 3e-4` on every rung, the muP base rate at
base width 256: hidden and readout matrices carry a weight multiplier
`256/w`, which under Adam scales their effective step by `256/w`, with init
chosen so effective weights start at `0.5/sqrt(fan_in)` on every rung;
embeddings, biases and norm gains keep the base rate. Head dim is fixed, so
attention logit scaling is unchanged.

```
$ python scaling.py --model hybrid --dry
size         params  kwargs
xs          111,019  {'width': 64}
s           840,657  {'width': 128}
m         2,804,091  {'width': 192}
l         6,616,489  {'width': 256}
xl       22,248,849  {'width': 384}
xxl      52,659,081  {'width': 512}
```

xxl, batch 64, bf16, forward+backward: **peak 14.23 GiB allocated (14.57
reserved)**, 0.39 s/iter on the local RTX PRO 4000 (23.4 GiB).

## Expected failure modes

* The decay gate must stay near 1 over ~330 tokens; it starts there and its
  gradient is small by construction (`sigmoid'(3)/16`), so until the k/v
  projections learn what to accumulate the state is a leaky average. After
  the 500-step smoke run the output is still nearly insensitive to the
  direction stream (mean |pred change| 3e-4 when directions are permuted).
* The sum lives in a 32x32 state per head read through `q`, and attention
  must match an offset embedding against it at pixel resolution; state size
  and offset-embedding resolution are the likely bottleneck, so the curve
  may bend if width grows too slowly.
* Each output patch straddles up to four input patches at an alignment
  inferred from the offset: a learned gather that may need depth more than
  width.
* muP transfer assumes 3e-4 is near-optimal at width 256; not tuned.

## Smoke results (local GPU)

`python train.py --model hybrid --size xs --steps 500`: 111,019 params,
floor mse 0.0969 / fg_mae 0.6298. Final eval **mse 0.0957**, fg_mae 0.6216,
fg_within_0.1 0.076, 58 it/s (9 s). Beats the floor, barely, like mlp
(0.0958 at 300 steps) and vanilla (0.0948 at 500 steps). Extra checks, not
deliverables: xs for 3000 steps reaches mse 0.0914 / fg_mae 0.597 (55 s,
108 it/s); s for 2000 steps reaches mse 0.0935 / fg_mae 0.608 (49 it/s).
Learning is slow in the first few thousand steps on every arm; the 20k-step
ladder on the cluster is where the comparison is made.
