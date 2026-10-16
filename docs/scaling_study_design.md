# Scaling study: compute vs capacity for hybrid4 and vanilla4 on SE(2)

Status: queued on the Torch cluster 2026-10-16. The predictions below were
written before any of these runs reported (see the commit time).

## Question

vanilla4 (all attention) solves the 32-step SE(2) task once it is big
enough. hybrid4 (Gated DeltaNet with beta in (0, 2), alternating with
attention) solves it much smaller and in fewer steps. Two readings:

- **Constant multiplier.** hybrid4 is more efficient by a constant compute
  multiplier, the same exponent shifted.
- **Different exponent.** The gap grows with scale or with task difficulty.

This study measures which, by separating capacity (parameters N) from
training (episodes D) and reporting loss against compute.

## Design

The task is SE(2), MNIST texture, per-step queries, batch 64, seed 0
unless noted. Every run uses warmup-stable-decay with checkpoints every
50k steps.

| Block | What | Sizes | Budget | Purpose |
| --- | --- | --- | --- | --- |
| A | long runs, 32-step walks | xs s m l xl, both models | 400k steps | the T=32 ladder; most already running |
| B | long runs, 64-step walks | xs s m l, both models | 400k steps | a harder task that stays off the floor; tests whether the gap grows with composition depth |
| C | cooldown branches from A and B | every long run | 62.5k, 125k, 250k steps | four data budgets per size from one long run; a branch equals a fresh run of that budget exactly (verified locally) |
| D | learning-rate check, 32 steps | s and l, both models | 125k steps at lr 1e-3 and 1e-4 | is the single muP base lr (3e-4) near-optimal for both, compared with the 125k branch at 3e-4? |
| E | second seed, 32 steps | s and m, both models | 400k steps | seed variance near the learning transitions |

Per model and walk length, that gives 4-5 sizes x 4 budgets = 16-20 loss
points from 4-5 long runs.

## Measurements and analysis

- **Loss.** All-query eval mse on the fixed held-out set (MNIST test
  digits), plus per-prefix bins and the last-step bin.
- **Compute, in two currencies.**
  - **FLOPs.** About 6 x N x tokens per episode x episodes, plus the
    attention term (sequence length 401 at T=32, 721 at T=64) and the
    delta-rule scan, counted per architecture and stated.
  - **H200 wall-clock,** from the logs.
- **Fit per model and walk length:** L(N, D) = E + A N^-alpha + B
  D^-beta (Chinchilla form). Read off the compute-optimal frontier.
- **Headline: the effective compute multiplier.** At matched loss, how
  many times more compute vanilla4 needs than hybrid4, plotted against the
  loss level. A roughly flat ratio means a constant multiplier. A ratio
  that grows as loss falls means a different exponent.
- **Threshold view.** The compute at which each model first meets
  "solved" (all-query at most 0.030 and last-step at most 0.040), per
  size: the capability view of the same data.

Caution: at T=32 the larger models reach errors near 1e-4, where the fit
is dominated by noise and the eval set's resolution. The fits use only
points well above that, and T=64 exists to keep the curves informative.

## Predictions (registered before results)

- **S1.** At T=32, hybrid4's effective compute multiplier over vanilla4
  is above 1 at every loss level reached by both.
- **S2.** The multiplier grows as loss falls: hybrid4's compute exponent
  is larger. Uncertain; this is the interesting outcome.
- **S3.** At T=64 the multiplier at matched loss is larger than at T=32,
  because composition depth doubles.
- **S4.** Block D: neither model's best lr is more than 3x from the base,
  and at each size the base is within 1.5x of the best loss. If this
  fails, the single-lr ladders cannot support an exponent claim.

## Compute estimate

From the H200 benchmark (batch 64) and a local timing at T=64. Steps/s at
T=32 are hybrid4 xs/s/m/l/xl ~70/53/38/28/15 and vanilla4 ~100/95/69/56/28.
T=64 is about 2x slower.

| Block | Jobs | GPU-hours (approx) |
| --- | --- | --- |
| A (new: hybrid4 xl) | 1 | 8 |
| B | 8 | 55 |
| C | 54 | 25 |
| D | 8 | 6 |
| E | 4 | 8 |
| total new | 75 | ~100 |

At 3-4 concurrent GPUs that is about 1-1.5 days. Branches wait on their
parent long runs (afterany dependency plus a check that the checkpoint
exists).

## Not in this study

xl at T=64 (about 20+ hours per run; add if the T=64 ladder shows a
widening gap), the learning-rate grid at every size, and more seeds.
