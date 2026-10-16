# hybrid4.5b: a fixed pose coprocessor beside a growing transformer trunk

Lift notes section 03, form "b". Code: `mapping/models/se2/hybrid45b.py`
(coprocessor first) and the placement variants `hybrid45b_mid`,
`hybrid45b_last`, `hybrid45b_two` and `hybrid45b_nocopro` (the minimal-pair
control). Tests: `tests/test_prefreeze.py`.

## Architecture

- **Trunk.** vanilla4 at the rung, exactly. The class subclasses `Vanilla4`
  and calls its constructor, so embeddings, query placement, RoPE on the
  step index, the w/32 attention + SwiGLU blocks, the muP multipliers and
  the initialisation are vanilla4's, with the same parameter names.
- **Coprocessor.** `x <- x + up(B64(down(LN(x))))`. `B64` is hybrid4 xs's
  first block, built with hybrid4 xs's arguments: a pre-norm bidirectional
  Gated DeltaNet with 2 heads of 32 (one per scan direction), β in (0, 2),
  a causal short conv of width 4 per scan direction, then SwiGLU (hidden
  192). It has its own residual stream. The conv uses the muP-governed init
  of `hybrid4_muconv`. The block is the same at every rung.
- **Projections, muP.**
  - `down` (w -> 64) is output-like, so it follows the trunk's hidden rule:
    multiplier 256/w, effective std 0.5/sqrt(w).
  - `up` (64 -> w) is input-like, so its learning rate must not shrink with
    w. It uses the coprocessor's multiplier (4), effective std 0.5/8, and
    the trunk's residual factor 1/sqrt(2 · trunk blocks).
- **Placement.** For each coprocessor, the number of trunk blocks that run
  before it: first = 0, middle = n/2, last = n (after every attention
  block, just before the final norm and readout).
- **Initial values.** Coprocessors are built after the trunk, so at a given
  seed the trunk's initial values are vanilla4's. The pair therefore
  differs only by the coprocessor, at initialisation too. Tests assert
  this, and assert that `hybrid45b_nocopro` is vanilla4 (same state dict,
  same outputs).

| size | width | trunk blocks | vanilla4 = nocopro | hybrid45b (one copro) | hybrid45b_two |
| --- | --- | --- | --- | --- | --- |
| xs | 64 | 2 | 114,441 | 182,317 | 250,193 |
| s | 128 | 4 | 822,281 | 898,541 | 974,801 |
| m | 192 | 6 | 2,688,777 | 2,773,421 | 2,858,065 |
| l | 256 | 8 | 6,476,297 | 6,569,325 | 6,662,353 |
| xl | 384 | 12 | | 21,445,613 | |
| xxl | 512 | 16 | | 50,886,253 | |

One coprocessor is 59,428 parameters (the block) plus 129w + 64
(projections and LayerNorm).

## Mechanism trace

1. **The token geometry is hybrid4's.** The coprocessor runs on the placed
   sequence: obs_t ×9, then queries at t, then odometry t. Its scan and conv
   follow sequence order only. So the step period, the 3-of-9 patches that
   see the odometry token through the width-4 conv, and the right-to-left
   span are all unchanged.
2. **The odometry symbol reaches k and β.**
   - At "first", the input is LN(embedding) through a random w -> 64 map.
     The 16 odometry embeddings stay separable in 64 dimensions.
   - At "middle" or "last", the input is the residual stream after
     attention. It still carries the embedding.
3. **One head per direction, 32×32 state.** This is the configuration in
   which hybrid4 xs learned its turn-locked rotation.
4. **The read-out is positional.** The coprocessor writes into every
   token's residual. Attention after it can match content against that
   frame. At "last" no attention follows, which is what the placement
   sweep tests.

## Audit

- **Trunk.** vanilla4's items apply unchanged.
- **Coprocessor.** hybrid4's items apply: the conv and scan mix tokens by
  sequence order only, and β, the decay and the conv are learned from token
  content.
- **Projections.** `down` and `up` are learned per-token linear maps. No
  odometry arithmetic happens outside learned layers.

## Probing

`tools/probe_gdn.py <ckpt>` finds the coprocessor (`model.gdn_layers()`)
at any placement and reports per head:
- β by token role and by patch position after the odometry token
- the turn-locked rotation rate (transition probe)
- crispness

Results: docs/lift/prefreeze_ablations.md.
