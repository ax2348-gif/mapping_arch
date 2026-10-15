# vanilla: the 2026-default transformer (the arm under interrogation)

**Architecture.** One flat token sequence, one pre-norm transformer, full
bidirectional attention (offline setting), queries as extra tokens in the
same stack. Blocks are RMSNorm -> multi-head attention (QK-norm, RoPE over
the sequence index, `F.scaled_dot_product_attention`) -> RMSNorm -> SwiGLU
(hidden 8/3 width, rounded to 64). No biases, no dropout, GPT-2 init
(std 0.02; residual output projections 0.02/sqrt(2 depth)). Final RMSNorm
and a linear head from width to 81 pixels on each query token.

**Tokens (337, fixed).** `[obs_0 x9][dir_0][obs_1 x9][dir_1] ... [dir_31]
[obs_32 x9][query_0..7]`.
- Observation: nine 3x3 patches (ViT-style), each a linear map of its 9
  pixels plus a learned patch-index embedding. Chosen over a frame-level
  token because it is the standard image tokenisation, keeps the token
  content small, and gives attention something finer than a window to
  point at; it is also the 3x3 grid of the related marlenv work.
- Direction: learned embedding of the 0..3 index, placed between the two
  windows it joins. Never summed or indexed outside learned layers.
- Query: Fourier features of the raw (dy, dx) offset / 288 at nine octaves
  (full range down to ~2 px) through a two-layer MLP, the usual recipe for
  continuous scalar conditioning. A learned 577-entry table per axis would
  also be legal; the Fourier form gives ordinal structure between
  neighbouring offsets for free, which matters at pixel resolution.
- A learned token-type embedding (patch / direction / query) on every token.

**Readout.** Query tokens ride in the same sequence and the same layers
(MAE-decoder / BERT-[MASK] pattern) and are decoded from their own final
state. Picked over a separate cross-attention decoder (Perceiver-IO) as the
more standard option: one stack of weights, and queries take part from
layer one instead of only after the encoder has finished.

**Positions.** RoPE over the flat sequence index (obs t patch p at 10t+p,
direction t at 10t+9, queries at 329..336), a learned per-step embedding
(33 entries, by time step, added to the patches of observation t and to
direction token t: the ViT/BERT absolute convention alongside RoPE's
relative one), and the within-observation patch index. Nothing is keyed on
directions or offsets; no canvas, no cumulative sums, no scatter/gather.
The full audit is in the module docstring.

**Why it might not integrate the directions.** The model has no mechanism
for summing moves except what attention and MLPs learn: a query must
discover that "the window at offset (dy, dx)" is the step whose suffix of
direction tokens sums to (dy, dx), a prefix-sum-then-match computation
that soft attention over a flat sequence has to build in depth. Expected
failure modes: a long plateau just off the floor (it learns the mean
digit statistics first), then slow improvement driven by depth more than
width, and a curve that bends early compared with any variant that
carries an integration bias.

**Scaling rule.** Head dim 64; heads = width/64; depth = heads + 2; hidden
= 8/3 width. One parameter (width); the +2 is a depth floor so xs has four
layers rather than two (max(4, heads) was rejected only because it leaves
a 96x span, under the contract's two orders). Learning rate: standard
parameterisation and the harness 3e-4 AdamW (warmup + cosine) at every rung,
inside the usual stable range for widths 128..768 (GPT-2/3 used 6e-4 at
768 and 3e-4 at 1024). muP-style transfer (lr ~ 256/width) is the stated
alternative; the harness has a single `--lr`, so it would need one
`scaling.py --sizes <rung> --extra "--lr ..."` call per rung.

```
$ python scaling.py --model vanilla --dry
size         params  kwargs
xs          892,881  {'width': 128}   (2 heads, 4 layers)
s         2,285,713  {'width': 192}   (3, 5)
m         4,932,177  {'width': 256}   (4, 6)
l        14,379,217  {'width': 384}   (6, 8)
xl       32,478,033  {'width': 512}   (8, 10)
xxl      99,841,105  {'width': 768}   (12, 14)
```

**Largest rung.** xxl, batch 64, bf16 autocast, forward + backward + AdamW
step on the local RTX PRO 4000 (23.4 GiB): peak allocated 14.04 GiB, peak
reserved 14.60 GiB, 475 ms/step (2.1 it/s). A width-1024 rung (18 layers,
~240M params) would not fit, so the ladder stops at 768.

**Smoke results (local GPU, seed 0, batch 64).**
`python train.py --model vanilla --size xs --steps 500`:
floor mse 0.0969 / fg_mae 0.6298; step 250 eval mse 0.0957, fg_mae 0.6007,
fg<0.1 0.088, dirs-shuffled mse 0.0957; **step 500 eval mse 0.0948, fg_mae
0.6157, fg<0.1 0.077, dirs-shuffled mse 0.0948, 42 it/s** (GPU shared with
another job during the run). Beats the floor at 500 steps, by a little, as
the reference mlp does; the shuffled-directions number equals the plain
one, so at 500 steps it is not yet using the directions at all. With the
earlier depth = heads ladder (2 layers at xs, 462k params) a 3000-step xs
run kept moving: eval mse 0.0951 / 0.0938 / 0.0926 / 0.0913 / 0.0909 /
0.0908 at steps 500..3000; m reached 0.0946 at 500 steps and xl 0.0947 at
300 steps at the fixed lr with no divergence. Those runs predate the depth
floor and the per-step embedding and were not repeated.
