# vanilla2: the second vanilla arm (other standard option at every fork)

**What it is.** The counterpart of `vanilla`. Same 2026-default block
(pre-norm RMSNorm -> multi-head attention with QK-norm and
`scaled_dot_product_attention` -> RMSNorm -> SwiGLU at 8/3 width; no
biases, no dropout, GPT-2 init with residual projections scaled by depth),
full bidirectional attention, nothing that knows about space. Every
interface fork where `vanilla` picked one standard option, this arm picks
the other, so the two bracket the design space of "a careful engineer's
standard transformer" instead of reporting one draw from it.

**Forks, `vanilla` -> `vanilla2`.**
1. *Tokenisation.* Nine 3x3 patch tokens per window -> **one token per
   window**: the 81 pixels through a single linear embed (ViT with patch
   size = image). The encoder sees 33 tokens, not 329.
2. *Direction placement.* Separate direction token -> **fused into the
   observation it leads to**: direction t is added to token t+1 as a
   learned embedding; token 0 gets a learned "no move" entry. Additive
   rather than concatenated because under a linear embed the two are the
   same model (`W[x; e_d] = W_x x + w_d`); additive is the BERT-segment /
   ViT pattern and makes that explicit.
3. *Positions.* RoPE on the sequence index -> **learned absolute per-step
   embeddings** (33-entry table, BERT/ViT style), keyed on step index only.
4. *Readout.* Queries as extra encoder tokens (MAE/[MASK]) -> **cross-
   attention decoder** (Perceiver IO / DETR / T5): 2 pre-norm blocks of
   [self-attn among queries, cross-attn to the normed encoder output,
   SwiGLU], then RMSNorm and a linear head to 81 pixels. Queries never
   enter the encoder. Query self-attention is kept as the standard decoder
   block; the queries are independent so it should matter little.
5. *Offset encoding.* Fourier features + MLP -> **learned table**, one
   577-entry table per axis over [-288, 288], summed (factorised 2-D
   position tables are summed in the vision literature). No ordinal
   structure between neighbouring offsets: each entry is learned alone.
6. *Ladder.* Head dim 64, width the one parameter, but with a **depth
   floor**: depth = max(4, width/64), decoder depth fixed at 2. Widths
   64..768 give 0.44M..105M params, matching `vanilla`'s 0.46M..86M in
   parameters; per step each rung is about 10x cheaper than `vanilla`.
7. *Learning rate.* Single 3e-4 -> **per-rung rule** `LR[size] = 3e-4 *
   256 / width` (1.2e-3 .. 1e-4): lr inverse in width, the muP
   hidden-matrix rule applied to all parameters, equal to the harness
   default at width 256. `train.py` reads it when `--lr` is not given.

No fork needed a nonstandard substitute. The only judgement call is the
xs rung at width 64 (one head, 4 layers), which the depth floor plus the
parameter-matching requirement forces; it is a legitimate if small rung.

**Why it might not integrate the directions.** Same reason as `vanilla`:
nothing sums moves except what attention and MLPs learn. Here the
direction is content inside the step token, so a query must learn, in the
decoder's cross-attention, to match an offset-table entry against step
tokens whose encoder state has (somehow) accumulated the suffix of moves,
with absolute step embeddings as the only positional handle. Expected
failure modes: long plateau at the mean-digit solution, then slow gains;
the learned offset table may lag Fourier features at pixel resolution.
The shuffled-directions eval (below) shows the odometry is unused in the
smoke runs, as for `vanilla`.

**Audit** (full text in the module docstring): step table keyed on step
index; direction and "no move" embeddings as token content; offset tables
indexed by the raw offset only; cross-attention with no positional bias;
fixed row-major flatten of the window. No RoPE, no relative bias, no
cumulative sums, no canvas, no scatter/gather, nothing reads `positions`.

```
$ python scaling.py --model vanilla2 --dry
size         params  kwargs
xs          441,105  {'width': 64}
s         1,585,617  {'width': 128}
m         5,693,265  {'width': 256}
l        15,864,785  {'width': 384}
xl       34,917,457  {'width': 512}
xxl     104,876,369  {'width': 768}
```

**Largest rung.** xxl, batch 64, bf16 autocast, forward + backward + clip
+ AdamW step on the local RTX PRO 4000 (23.4 GiB): peak allocated 2.95
GiB, peak reserved 3.05 GiB, 46 ms/step. Memory is dominated by the
105M-parameter optimizer state, not activations (33 tokens per episode).

**Smoke results (local GPU, seed 0, batch 64, lr 1.2e-3 from `LR`).**
`python train.py --model vanilla2 --size xs --steps 500`: floor mse 0.0969
/ fg_mae 0.6298; step 250 eval mse 0.0970, fg_mae 0.6327, fg<0.1 0.070;
**step 500 eval mse 0.0958, fg_mae 0.6214, fg<0.1 0.076, 116.6 it/s**;
dirs-shuffled mse 0.0958 at both evals (equal to unshuffled: odometry not
used yet). Beats the floor at 500 steps, by a little, like `vanilla`
(0.0948) and the reference mlp. A 3000-step xs run (about 20 s): eval mse
0.0959 / 0.0958 / 0.0950 / 0.0937 / 0.0933 / 0.0931 at steps 500..3000,
fg_mae 0.6055, fg<0.1 0.081, shuffled 0.0931 at the end, 179 it/s.
Stability at the per-rung lr: m (3e-4) reaches 0.0957 at 500 steps (113
it/s); xl (1.5e-4) reaches 0.0958 at 300 steps (42 it/s); no divergence.
