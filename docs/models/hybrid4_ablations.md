# Ablations of hybrid4

Each is hybrid4 with one change. All were run on SE(2), xs, MNIST,
per-step queries, 100k steps, seed 0. The floor is 0.0986.

| model | change | all-query mse | last-step | shuffled | verdict |
| --- | --- | --- | --- | --- | --- |
| hybrid4 | none | 0.0180 | 0.0263 | 0.1403 | solved (seed 1: 0.0180) |
| `hybrid4_gelu` | ungated 4x GELU MLP instead of SwiGLU | 0.0416 | 0.0693 | 0.1024 | in between, still improving |
| `hybrid4_beta1` | beta = sigmoid, in (0, 1): no reflections | 0.0491 | 0.0718 | 0.0918 | failed |
| `hybrid4_noconv` | no short convolution (SiLU kept) | 0.0537 | 0.0742 | 0.0884 | failed |
| `hybrid4_cdiag` | commuting complex-diagonal transition: learned angle per channel pair, additive write, no erase | 0.0563 | 0.0748 | 0.0874 | failed |
| `hybrid3` | gated linear attention instead of the delta rule | 0.0549 | 0.0731 | 0.0892 | failed |

**What the mechanism probes show.**
- **hybrid4.** The right-to-left head rotates a 2-D subspace by exactly
  the turn on every step, and the rotation composes over 16 steps.
- **hybrid4_beta1.** There is no rotation pair on any turn step, and
  beta is pressed against its cap of 1 on turn tokens.
- **hybrid4_cdiag.** It learned a fixed step clock of about 20-27
  degrees per step, with 0 of 16 channel pairs locked to the turn. It
  had the capacity to rotate by the turn but did not learn to. It also
  differs from hybrid4 in three ways at once (commuting, no erase,
  angle parametrisation).

Details, dates and caveats are in the amendments of HYPOTHESIS.md. The
probe tools are in tools/.
