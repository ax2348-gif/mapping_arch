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

## Submitted (2026-10-16)

| Block | Job ids |
| --- | --- |
| A | hybrid4 xl 20028840; already running or queued: vanilla4 l, xl 20014708; hybrid4 l 20014709; vanilla4 xs, s, m 20028595; hybrid4 xs, s, m 20028596 |
| B | hybrid4 T=64 20028842; vanilla4 T=64 20028843 |
| C | branches 20028849-20028855, each gated (afterany) on its parent array |
| D | hybrid4 20028844; vanilla4 20028846 |
| E | hybrid4 seed 1 20028847; vanilla4 seed 1 20028848 |

## Cancelled to prioritise the lift (2026-10-16, user's call)

The lift's Torch jobs (worker 20038588, m bridge 20038403) sat behind our own
older pending scaling jobs. Users here cannot reorder jobs, so every PENDING
scaling task was cancelled. Running tasks were untouched, and array 19991387
stays. To resubmit, from /scratch/ax2348/mapping. The job files are unchanged
there; the lift runs from the separate tree /scratch/ax2348/mapping_lift.

| was | what | resubmit |
| --- | --- | --- |
| 20028842_[2-3] | hybrid4 T=64, m and l | `sbatch --array=2-3%3 jobs/se2_hybrid4_400k_qt_wsd_t64.sbatch` |
| 20028843_[2-3] | vanilla4 T=64, m and l | `sbatch --array=2-3%3 jobs/se2_vanilla4_400k_qt_wsd_t64.sbatch` |
| 20028844_[2-3] | hybrid4 lr check, l at 1e-3 and 1e-4 | `sbatch --array=2-3%2 jobs/se2_hybrid4_125k_lrcheck.sbatch` |
| 20028844_1 (RUNNING, stopped ~35 min in to free a GPU for the lift worker) | hybrid4 lr check, s at 1e-4 | `sbatch --array=1 jobs/se2_hybrid4_125k_lrcheck.sbatch`; its partial run dir should be cleared first |
| 20028846_[1-3] | vanilla4 lr check: s at 1e-4; l at 1e-3 and 1e-4 | `sbatch --array=1-3%2 jobs/se2_vanilla4_125k_lrcheck.sbatch` |
| 20028849_[0-8] | hybrid4 T=32 branches, xs s m | `sbatch --dependency=afterany:20028596_1 --array=0-5%3 jobs/cd_h4_t32_small.sbatch` (xs, s, after the s parent), and `--array=6-8%3` after the m parent (20028596_2 was also pending, below) |
| 20028850_[0-8] | vanilla4 T=32 branches, xs s m | `sbatch --array=0-5%3 jobs/cd_v4_t32_small.sbatch` (parents done), and `sbatch --dependency=afterany:20028595_2 --array=6-8%3 jobs/cd_v4_t32_small.sbatch` |
| 20028851_[0-5] | vanilla4 T=32 branches, l xl | `sbatch --dependency=afterany:20014708_1 jobs/cd_v4_t32_large.sbatch` |
| 20028852_[0-2] | hybrid4 T=32 branches, l | `sbatch --dependency=afterany:20014709_0 jobs/cd_h4_t32_l.sbatch` |
| 20028853_[0-2] | hybrid4 T=32 branches, xl | `sbatch --dependency=afterany:20028840_0 jobs/cd_h4_t32_xl.sbatch` |
| 20028854_[0-11] | hybrid4 T=64 branches | `sbatch --dependency=afterany:<T=64 parents> jobs/cd_h4_t64.sbatch` (the m and l parents must be resubmitted first) |
| 20028855_[0-11] | vanilla4 T=64 branches | `sbatch --dependency=afterany:<T=64 parents> jobs/cd_v4_t64.sbatch` |
| 20028596_2 | hybrid4 T=32 m, the long run | NOT cancelled: it started a few minutes before the cancel and is running |

Each branch task first checks that its checkpoint exists, so a branch whose parent did not reach it fails fast rather than running from nothing.
