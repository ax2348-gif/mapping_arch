# hybrid4: design and working principle

hybrid4 is the first model in this study to learn the SE(2) mapping task.
At its smallest rung (xs, 119,661 parameters) it reached eval mse 0.0180
against a constant floor of 0.0986 after 100k steps with per-step queries.
This page explains how it works, from tokens to output, and why each part
is there. Code: `mapping/models/hybrid4.py` (Z²) and
`mapping/models/se2/hybrid4.py` (SE(2)); the two differ only at the input
interface (section 2).

Where this page says "the model does X", it describes what the
architecture makes cheap. We have not yet probed a trained model's
internals, so statements about what it *actually* learned are marked as
hypotheses.

---

## 1. The task, briefly

An agent walks 32 steps of 9 pixels over a map of MNIST tiles and sees a
9×9 window at each of its 33 positions. It is then asked to render the 9×9
window at a query pose. Every pixel of the answer was seen at some earlier
step; the model's job is to work out *which* earlier pixels, from the
odometry alone.

- **Z²:** the agent only translates. Odometry is one of four directions
  per step. Placing an earlier window relative to the current one means
  summing the moves in between.
- **SE(2):** the agent also has a heading, drawn at random each step. Each
  window is shown rotated so the agent's heading is "up". Odometry is the
  move *in the agent's own frame* (forward, right, back, left) plus the
  turn to the next heading, as one of 16 symbols. Placing an earlier
  window now means composing poses: every move must be rotated by the
  heading accumulated since, and the heading is itself a running sum of
  turns. A prefix sum is no longer enough, and the fragments arrive in four
  orientations that must be aligned.
- **Per-step queries:** each of the eight queries in an episode is asked
  at a random step t (0 to 32), relative to the pose at t, and its answer
  only uses observations up to t. Short-t queries need few compositions,
  long-t queries need many, so one episode spans easy to hard.

## 2. Inputs and tokens

The model receives the 33 observations, the 32 odometry symbols, and for
each query its offset (Z²: `(dy, dx)`; SE(2): body-frame `(dy, dx)` plus a
heading change 0–3) and its step t. Everything becomes a sequence of
vectors of width w.

| token | how it is made | count |
| --- | --- | --- |
| observation patch | each 9×9 window is cut into nine 3×3 patches; a patch's 9 pixels go through a linear map, plus one of 9 learned **patch-position** vectors (which of the nine patches it is). The same 9 vectors are used at every step | 9 per step, 297 |
| odometry | a learned vector per symbol (4 on Z², 16 on SE(2)) | 1 per step, 32 |
| query patch | the query offset as Fourier features (the two scaled numbers plus sine and cosine at 8 octave frequencies each, 34 features) through a linear map, plus one of 9 learned **query-position** vectors (which output patch this token will draw). On SE(2) a learned vector for the heading change (one of 4) is added too | 9 per query, 72 |

Total: 401 tokens. Nothing in a token says *which step* it belongs to.
Step information enters in exactly two ways: the order of the sequence,
and RoPE in the attention layers (section 4). This is deliberate. Once
attention can tell steps apart by rotation, separate per-step tables are
redundant, and dropping them forces the model to use relative step
structure rather than memorising absolute steps.

### Sequence order, and why queries sit at their own step

```
obs_0 (9 tokens)  dir_0  obs_1 (9)  dir_1  ...  obs_t (9)  [query asked at t (9)]  dir_t  ...  obs_32 (9)
```

A query asked at step t has its nine tokens placed directly after
observation t and before odometry token t. Several queries at the same
step sit together, in query order. With every query at the last step this
is simply "queries at the end", the original layout.

The placement matters because of the recurrent layer (section 3). A
left-to-right scan reaching a query token has read exactly the
observations and moves up to the query's own step, which is the frame the
query is expressed in. Put the query at the end instead and its scan state
describes the end of the episode, not step t; that version stalled (the
"end-placed" result in the study log). Placement costs nothing: it is a
fixed reordering computed from the query step, a sequence index, and the
outputs are put back in their original order before the readout.

## 3. The recurrent layer: bidirectional Gated DeltaNet

This is the layer that made SE(2) learnable. It is a linear recurrence
with a matrix-valued state per head.

### 3.1 The state and its update

Each head keeps a state matrix S of size 32×32. Think of S as a small
associative memory: writing pairs a 32-dim key with a 32-dim value, and
reading with a query returns roughly the value stored under the matching
key. At every token the state is updated:

```
S  ←  a · (I − β k kᵀ) · S  +  β k vᵀ
o  =  Sᵀ q
```

Here k, v and q are computed from the token, with k scaled to unit length.
The three scalars per token and head:

- **a, decay:** `exp(logsigmoid(z) / 16)`, between 0 and 1, initialised
  near 0.997. It lets the head forget slowly or quickly depending on the
  token.
- **β, write strength:** `2 · sigmoid(z)`, between 0 and 2, initialised at 1.
- **The bracket `(I − β k kᵀ)`** acts only along the direction k.
  Whatever the state stores along k is scaled by (1 − β). Everything
  orthogonal to k is untouched.

Read the update as "erase, then write". With β = 1, the old content
along k is removed and the new value v is written there. This is the
delta rule: the state is corrected toward storing v under k, instead of
having v piled on top of what was already there as in plain linear
attention.

### 3.2 Why β up to 2 matters: reflections compose into rotations

With β = 2 the bracket becomes `I − 2 k kᵀ`, a **reflection**: it flips
the state's component along k and leaves the rest. Two reflections in a
row, about two different directions k₁ and k₂, make a **rotation**:

```
(I − 2 k₁k₁ᵀ)(I − 2 k₂k₂ᵀ)  =  rotation by twice the angle between k₁ and k₂
```

In the plane spanned by k₁ and k₂, two reflections at 45° to each other
rotate by 90°, a quarter turn. We checked this numerically for the layer:
two β = 2 updates give an orthogonal map with determinant 1.

This is exactly what SE(2) needs. Suppose some channels of the state hold
"where the window from step s is, relative to me, in my frame". When the
agent turns by 90°, every such stored vector must be rotated by 90° to
stay correct in the new frame. A plain gated linear attention, which the
earlier hybrids used, can only multiply the whole state by a positive
number: it can shrink stored vectors but never rotate them. Before the
change, the only way to handle turns was to improvise the rotation
elsewhere in the network, and every SE(2) run before hybrid4 stalled.

Restricting β to (0, 1], the usual setting, would allow erasing but not
reflecting, and so no rotation. The (0, 2) range is from Grazzi et al.
(2024), "Unlocking state-tracking in linear RNNs through negative
eigenvalues".

### 3.3 Why the short convolution is part of the mechanism

A rotation needs *two* reflections, and each token applies one update.
The odometry token for a step can supply one reflection, keyed by the
turn. A single reflection reverses orientation (determinant −1), so on its
own it cannot be a rotation. The second reflection has to come from a
neighbouring token, but the next tokens are observation patches, which
know nothing about the turn.

Before the scan, q, k and v pass through a depthwise convolution of width
4 along the sequence, followed by SiLU. "Depthwise" means each channel is
mixed only with the same channel at the 3 previous positions. After it,
the patch tokens right after an odometry token carry part of that token's
content, so they can produce keys and write strengths that depend on the
turn and supply the second reflection. This is a hypothesis about how the
trained model uses the convolution. The architectural point is that
without the convolution the second reflection has no source.

### 3.4 Two directions

Half the heads scan left to right and half right to left. Each half has
its own convolution, running in its own scan direction, so a left-to-right
head at position t sees only tokens up to t and a right-to-left head only
tokens from t on. This is checked by a test.

- **Left to right.** At a query token placed after observation t, the
  state summarises the walk up to t, in whatever frame the head maintains.
  This is what a query anchored at t needs.
- **Right to left.** At observation s, the state summarises everything
  after s. This gives each observation token a path to "where am I
  relative to later steps", useful for queries anchored later.

Using both is allowed because the task is offline: the whole episode is
available before any query is answered.

### 3.5 Output and kernel

The scan output per head is RMS-normalised, multiplied by a SiLU gate
computed from the layer's input (a GLU on the read-out), and projected
back to width w. The scan runs on `chunk_gated_delta_rule` from
flash-linear-attention 0.5.2, a fused Triton kernel. It is checked
against a plain PyTorch loop of the update in 3.1, kept as `delta_scan`
in `hybrid3_5.py`:

| check | relative error |
| --- | --- |
| outputs, fp32 | 2e-3 |
| outputs, bf16 | 7e-3 |
| every gradient | 2e-3 to 4e-3 |
| the whole model, with hybrid3.5's weights | 3e-4 |

The kernel is about 30 times faster per training step at xs.

## 4. The attention layer: softmax with RoPE on the step index

Every other block is ordinary multi-head softmax attention over all 401
tokens, with no mask. Queries and keys are rotated by **RoPE using each
token's step**: observation and odometry tokens of step t use position t,
and a query's tokens use their query step. Base 10000, 16 rotary pairs per
head of 32 dimensions.

RoPE makes the attention score between two tokens depend on their step
difference, not their absolute steps. We checked that shifting both steps
by the same amount leaves the score unchanged. From the first training
step, attention therefore has a notion of "same step" and "a few steps
apart". In hybrid it had to learn this from a randomly initialised
per-step table, and dropping that table in favour of RoPE (the hybrid2
change) made early learning much faster on Z².

The division of labour this suggests, as a hypothesis: the recurrent layer
computes *where* things are relative to an anchor, which needs
composition over many steps, and attention does content retrieval, which
fragment and which pixels, keyed on what the recurrence wrote into the
token states.

Attention runs on PyTorch's fused attention, which uses the FlashAttention
kernel. It is about 10% of a training step at every rung.

## 5. Blocks, MLP and readout

Each block is pre-norm: `x ← x + mixer(LayerNorm(x))`, then
`x ← x + MLP(LayerNorm(x))`. Mixers alternate, GDN first.

The MLP is **SwiGLU**: `down(SiLU(gate(x)) · up(x))`, hidden width 8/3 w
rounded up to a multiple of 32. Multiplying two linear maps of the input
lets a block compute products of its features directly, for instance
"turn feature × move feature", which a GELU MLP can only approximate.
Whether SwiGLU or the delta rule carries the SE(2) result is what the
hybrid3 run in progress measures.

**Readout.** After the last block, tokens are put back in the original
order. The 72 query tokens pass through a final LayerNorm and one linear
map to 9 pixels each, a 3×3 patch. The nine patches of a query are
reassembled into its 9×9 window by their patch position, the inverse of
the input patching. The readout starts at zero, so the untrained model
predicts all-black windows.

## 6. Sizes, parameters and training

One parameter, the width w, sets everything. Heads = w/32, half per
direction. Blocks = w/32, half GDN and half attention.

| rung | width | heads per direction | blocks (GDN + attention) | SwiGLU hidden | parameters (SE(2)) |
| --- | --- | --- | --- | --- | --- |
| xs | 64 | 1 | 1 + 1 | 192 | 119,661 |
| s | 128 | 2 | 2 + 2 | 352 | 860,505 |
| m | 192 | 3 | 3 + 3 | 512 | 2,813,901 |
| l | 256 | 4 | 4 + 4 | 704 | 6,768,329 |
| xl | 384 | 6 | 6 + 6 | 1024 | 22,306,137 |
| xxl | 512 | 8 | 8 + 8 | 1376 | 53,041,673 |

The SE(2) solve happened at **xs**: one delta-rule head per direction,
each with a 32×32 state, followed by one attention layer. Where the xs
parameters go:

| part | parameters |
| --- | --- |
| delta-rule mixer | 21,860 |
| attention mixer | 16,640 |
| two SwiGLU MLPs | 74,624 |
| embeddings, readout and norms | the rest |

**Learning rate and initialisation.** The hidden and output matrices are
"muP" linear layers: each weight is stored divided by m = 256 / w and
multiplied by m when used. Under Adam this scales those matrices'
effective learning rate by 256/w, so one base learning rate, 3e-4, works
at every width. Embeddings, biases and norm gains keep the base rate. The
effective weights start at standard deviation 0.5/√fan-in, and the
residual-branch outputs are scaled down by 1/√(2 · blocks).

**Training recipe for the result.** AdamW with betas (0.9, 0.95), weight
decay 0.1 on matrices, 100 warmup steps, cosine decay to zero, gradient
clipping at 1.0, batch 64, bf16 autocast, mean squared error on pixels.

## 7. What the result looks like

SE(2), xs, seed 0, per-step queries, eval mse by the step a query is
asked at:

| queries at steps | 20k | 40k | 60k | 100k |
| --- | --- | --- | --- | --- |
| 0 to 7 | 0.0190 | 0.0165 | 0.0137 | 0.0118 |
| 8 to 15 | 0.0566 | 0.0375 | 0.0242 | 0.0180 |
| 16 to 23 | 0.0695 | 0.0497 | 0.0289 | 0.0198 |
| 24 to 32 | 0.0779 | 0.0604 | 0.0336 | 0.0218 |
| last step | 0.0774 | 0.0708 | 0.0443 | 0.0263 |
| odometry shuffled | 0.0902 | 0.1090 | 0.1303 | 0.1403 |

Short compositions are learned first. Longer ones follow in a series of
drops, and the gaps narrow. Shuffling the odometry pushes the error well
above the floor, so the model depends on it. The grid shows rotated digit
fragments drawn in the right place and orientation. One seed, one rung so
far.

## 8. Lineage

| model | change from its parent | what it showed |
| --- | --- | --- |
| hybrid | gated linear attention alternating with attention; learned per-step tables | solved Z² at xs, stalled on SE(2) |
| hybrid2 | RoPE on the step index; per-step tables removed; queries placed at their step | faster early learning on Z² with per-step queries; SE(2) still stalled |
| hybrid3 | SwiGLU MLPs | being measured on SE(2) at 100k |
| hybrid3.5 | Gated DeltaNet with β in (0, 2) and a short convolution, as a hand-written loop | reference implementation |
| hybrid4 | hybrid3.5 on the fused kernel | first SE(2) solve |

## 9. Known issues and open questions

- **Stale text and dead code.** `hybrid4.py` still contains the old
  gated-linear-attention layer, its scan and the `CHUNK` constant, none of
  which are used. The scaling-rule string still says "gated linear
  attention". Both are cosmetic.
- **Convolution initialisation.** The short convolution uses PyTorch's
  default initialisation, not the muP rule the other matrices follow. This
  is untested across rungs.
- **Queries write into the scan.** Query tokens sit inside the sequence,
  so they also write into the delta-rule state that later tokens read, and
  queries can attend to each other. They carry no answer information, only
  their offset and step, so nothing leaks. It does mean one query's tokens
  can influence another's output.
- **Attribution.** hybrid4 differs from hybrid2 by both SwiGLU and the
  delta rule. The hybrid3 run separates the two. Which part of the delta
  rule matters, β above 1 or the convolution, would need further
  ablations: β capped at 1, and no convolution.
- **Mechanism unverified.** That the state actually rotates on turns is a
  hypothesis. A probe would settle it: read the state at each step and
  test whether a linear map recovers the relative pose, and whether turn
  tokens act on it as rotations.
