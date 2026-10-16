# vanilla4: hybrid4 with every mixer as attention

The transformer counterpart of `hybrid4`, differing in one factor: every
token mixer is softmax attention with RoPE on the step index. hybrid4
alternates bidirectional Gated DeltaNet with that attention. Everything
else is hybrid4's:

- patch tokens, with nine shared patch embeddings and nine shared query
  embeddings
- no per-step tables
- per-step query placement
- SwiGLU MLPs and pre-norm LayerNorm blocks
- the ladder (blocks = heads = width/32)
- the muP learning-rate rule and initialisation

It has about 4% fewer parameters per rung (an attention mixer has 4w²
weights, a Gated DeltaNet mixer about 5w²).

**Mechanism.** With attention only, a token's sequence position does not
matter, only its RoPE step. A query asked at step t carries step t, so
the step difference to every fragment is visible from the first layer.
Composing moves and turns into a relative pose must happen across the
attention and SwiGLU layers. That composition is what the comparison with
hybrid4 tests (prediction P1 in HYPOTHESIS.md).

**Status (2026-10-16).**
- **Built for both tasks** and passes the model tests.
- **Throughput.** H200 benchmark at batch 64: s 95, m 69, l 56, xl 28
  steps/s.
- **Queued.** First large runs on the cluster: l and xl, SE(2) per-step
  queries, 400k steps, warmup-stable-decay, a single learning rate. A
  failure there would not yet meet P1's falsification bar, which needs
  the learning-rate grid.
