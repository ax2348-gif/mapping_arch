# Frozen hypothesis: ego-motion composition needs a group-structured operator

Frozen 2026-10-16 at git tag `hypothesis-v1`. Do not edit the sections
above **Amendments**. Later changes go into that section, dated, so the
original claims stay checkable against the results that come after them.

## Motivation

Practitioners report that end-to-end transformer policies given a stacked
visual history with ego-motion make poor use of that history, and that
video world models with memory struggle to stay spatially consistent. In
related work (marlenv), feeding dead-reckoned world coordinates into RoPE
fixed the problem, which performs the pose composition outside the
network.

## The hypothesis

**H.** To use an egocentric history, a sequence model must re-express
earlier observations in the current frame. Under rotation plus
translation this is a composition of rigid transforms: every move must be
rotated by the heading accumulated since. Learning this from data, at
modest scale, requires a sequence operator whose state transition can
represent the motion group's action, in particular rotation of a stored
quantity driven by the input. Two widely used operators lack this:

- softmax attention, whose positional information is a fixed function of
  the sequence index (RoPE, learned tables)
- linear recurrences whose transition is a positive scalar or diagonal
  decay

Architectures built only from these fail, or scale far worse, even when
every needed pixel is in context.

**Corollary.** Under translation only (Z², an abelian group) a positive
scalar-decay recurrence suffices, because composition reduces to a sum.
The difficulty appears specifically when the group stops commuting.

## Definitions used by the predictions

All on the study's standard SE(2) evaluation set (MNIST test split, eval
seed 0, 16 batches of 64), with per-step queries, unless a prediction
says otherwise. Floor: constant-prediction mse 0.0986.

- **Solved.** All-query eval mse ≤ 0.030 and last-step-bin mse ≤ 0.040,
  with odometry-shuffled mse above the floor.
- **Failed.** All-query eval mse ≥ 0.045 at the end of the stated
  budget.
- **Standard recipe.** Batch 64, AdamW, base learning rate 3e-4 (or the
  model's own per-rung rule), cosine schedule to zero, a single seed
  unless stated.

## Predictions and what would falsify them

**P1. Vanilla transformers fail on SE(2).**
- **Covers.** `vanilla` and `vanilla2`, trained on SE(2) with per-step
  queries at every rung of their ladders, up to about 100M parameters.
- **Budget.** Up to 400k steps, with learning rates chosen per rung from
  the grid {1e-3, 3e-4, 1e-4}.
- **Prediction.** Neither solves SE(2).
- **Falsified if.** Any rung of either solves it within that budget.
- **Part of P1.** Porting each model to SE(2) changes its interface only,
  as for the hybrids.

**P2. Rotation capability is the active ingredient.**
- **Covers.** hybrid4 with β capped at 1, no negative eigenvalues, under
  the standard recipe at xs for 100k steps.
- **Prediction.** It fails on SE(2), while still solving Z²
  (all-query mse ≤ 0.020 at 20k).
- **Falsified if.** The capped model solves SE(2).

**P3. The Z² / SE(2) dissociation persists with scale.**
- **Covers.** hybrid3, with SwiGLU and a scalar-decay recurrence, at
  every rung up to xxl, 100k steps.
- **Prediction.** It fails on SE(2) at every rung, while hybrid-family
  models with the same recurrence solve Z².
- **Falsified if.** Any rung solves SE(2).

**P4. Seed robustness.** hybrid4 at xs solves SE(2) under the standard
recipe at 100k steps in at least 2 of 3 seeds (seeds 0, 1, 2).

**P5. Mechanism.**
- **Prediction.** In a trained hybrid4 that solves SE(2), the
  left-to-right delta-rule state read at a query token linearly decodes
  the relative pose between the query's anchor and earlier steps.
- **Test.** Ridge probes from the state to (dy, dx) and heading change,
  with R² ≥ 0.8 on held-out episodes for anchors up to 16 steps back.
- **Falsified if.** The probes stay at R² < 0.5 while the model solves
  the task. The relative pose would then not be represented where this
  hypothesis says.

**P6. Transformer variants: exploratory, weaker prediction.**
- **Prediction.** Among one-change forks of vanilla, the one whose
  positional transform is input-dependent and orthogonal (PaTH
  attention) narrows the SE(2) gap the most. Sigmoid attention, weight-
  tied recurrent depth and Canon layers each narrow it less.
- **Falsified if.** Any of the latter three solves SE(2) under the P1
  budget while PaTH does not.

## Evidence at the time of freezing

SE(2), xs, seed 0, per-step queries unless noted:

| model | what it adds | steps | all-query mse | last-step mse | shuffled | status |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid (last-step queries) | scalar-decay recurrence + attention | 20k | 0.0863 | | 0.0892 | failed |
| hybrid2 | + RoPE on the step index, queries at their step | 20k | 0.0720 | 0.0841 | 0.0821 | failed |
| hybrid3 | + SwiGLU | 100k | 0.0549 | 0.0731 | 0.0892 | failed |
| hybrid4 | + Gated DeltaNet, β in (0, 2), short convolution | 100k | 0.0180 | 0.0263 | 0.1403 | solved |

On Z², the scalar-decay hybrid solved the task at xs (0.0150 at 20k,
last-step queries; 0.0115 with per-step queries). `vanilla`, `vanilla2`
and an LSTM drifted near 0.07 at 20k on Z² at xs. Vanilla has not been run
on SE(2).

Audit of the hybrid4 checkpoint (`tools/audit_se2.py`):
- episodes in a batch are isolated
- a fresh 4x test set gives 0.0174
- only 5.8% of targets are trivial copies of an observation
- randomising the turns, zeroing the turns, or randomising the moves
  each raises the error to about 0.13, above the floor, in every
  prefix-length bin
- randomising the query's heading change gives 0.109

Code at the freeze: model files at commits 27e112f (hybrid), 2ffe082
(hybrid2), 210442b (hybrid3), bb2aceb (hybrid4); harness and tests at
the tag.

## Known weaknesses at the time of freezing

- **Coverage.** One seed and one rung for the positive result.
- **Vanilla on SE(2).** It is untested, so the central contrast is
  inferred from Z² and from the hybrid lineage.
- **Distance from practice.** Perfect odometry, a discrete heading group
  (Z4), a 33-frame horizon, MNIST texture, and an offline setting. Real
  robots have continuous SE(3), noisy odometry and long histories, and
  video world models generate rather than retrieve.
- **Confounds in the lineage.** hybrid2 and hybrid4 differ by two
  changes: SwiGLU and the delta rule. The hybrid4_gelu ablation and P2
  address the split. Inside the delta rule, β above 1 and the short
  convolution are not yet separated.
- **The budget for "failed" is finite.** A model that fails at 400k steps
  could solve at 10M; P1's budget is a practical bound, not a proof of
  impossibility.

## Amendments

### 2026-10-16: seed replication and the SwiGLU ablation

New evidence, SE(2), xs, per-step queries, 100k steps, standard recipe:

| run | all-query mse | last-step mse | shuffled | status |
| --- | --- | --- | --- | --- |
| hybrid4, seed 0 | 0.0180 | 0.0263 | 0.1403 | solved |
| hybrid4, seed 1 | 0.0180 | 0.0225 | 0.1361 | solved |
| hybrid4_gelu (delta rule, GELU MLP), seed 0 | 0.0416 | 0.0693 | 0.1024 | neither solved nor failed |
| hybrid3 (SwiGLU, scalar-decay recurrence), seed 0 | 0.0549 | 0.0731 | 0.0892 | failed |

- **P4 is met: 2 of 2 seeds solve.** The trajectories agree to within
  0.0013 at every 20k checkpoint. Seed 1 is evaluated on its own eval set
  (eval seed = training seed; floor 0.0969), so the agreement is also
  across test sets. A third seed cannot change the outcome.
- **The delta rule works alone, but slowly.** Removing SwiGLU leaves a
  model that keeps improving across every prefix bin through 100k steps.
  Removing the delta rule (hybrid3) flattens by 40k. This is consistent
  with H: the rotation-capable operator is necessary at this budget, and
  the gated MLP sets how fast it is used. It does not test P2, which
  caps beta at 1.

### 2026-10-16: vanilla solves Z² with more steps (cluster results)

The frozen evidence section reports vanilla, vanilla2 and the LSTM
"drifted near 0.07 at 20k on Z² at xs". With more steps, vanilla solves
Z² completely (last-step queries, xs, lr 3e-4 unless noted):

| run | params | 20k | 40k | 300k |
| --- | --- | --- | --- | --- |
| vanilla xs | 0.9M | 0.0692 | 0.0110 | 0.0007 |
| vanilla m | 4.9M | | 0.0061 (lr 1e-4) | 0.0003 |
| vanilla, width 256, depth 32 | 25.8M | | 0.0025 | |
| vanilla2 s | 1.6M | | 0.0120 | |
| hybrid xs | 0.1M | 0.0150 | 0.0036 | |

- **The Z² gap is speed, not capability.** The hybrid reaches each level
  in fewer steps and parameters, but vanilla gets there. Depth helps
  strongly at a fixed step budget: 0.027 at depth 4 against 0.0025 at
  depth 32, at 40k.
- **H is unaffected, since it concerns SE(2).** The original hybrid on
  SE(2) with last-step queries stays near 0.070 at 400k steps under both
  cosine and warmup-stable-decay schedules.
- **P1 must be judged at a budget large enough.** Vanilla's Z² curve
  looks hopeless at 20k and solves by 300k, so a "vanilla fails SE(2)"
  finding needs a budget at least comparable, and the 400k bound in P1
  is the minimum, not a generous one.

### 2026-10-15: date correction

Dates written as 2026-10-16 in this file, in the header and the two
amendment headings above, should read 2026-10-15. The tag
`hypothesis-v1` and the commit timestamps are authoritative.

### 2026-10-15: the trained delta-rule state rotates with the agent

`tools/read_beta.py` and `tools/transition_probe.py` on the hybrid4 SE(2)
xs checkpoints, seeds 0 and 1. These measure the first-layer Gated
DeltaNet on 256 test episodes, on CPU.

- **Write strength.** Training moves beta well above 1. In the
  right-to-left head, every odometry token gets beta = 2.00, a pure
  reflection, with decay exactly 1.0000.
- **Per-step transition.** The homogeneous state update multiplied over
  each walk step (the odometry token plus the following observation
  patches, in that head's scan order) is, in the right-to-left head:
  - no turn: every significant eigenvalue at 0 degrees;
  - a quarter turn: exactly one conjugate pair at 90.0 to 90.3 degrees
    on 100% of steps, with |lambda| 0.998;
  - turning around: exactly two eigenvalues at 180 degrees on 100% of
    steps.

  The remaining ~24 significant eigenvalues stay at 0 degrees. So the
  head maintains one two-dimensional subspace that is rotated by exactly
  the agent's turn, and leaves the rest untouched.
- **The other direction.** The left-to-right head contracts its state
  over every step, with no eigenvalue above 0.5, so it acts as a
  short-memory head, not a pose tracker.

This is mechanistic support for H: the solving model represents the
heading change as a rotation of its state. It is not P5 as written, which
specifies a linear probe of relative pose. It does not show that the
rotation is necessary; P2 (beta capped at 1) tests that.

### 2026-10-15: P5 targets the wrong head; P5′ added (post hoc)

P5 as frozen names the left-to-right delta-rule state read at a query
token. The transition probe (previous amendment) found that in the solving
hybrid4 models the left-to-right head contracts its state over every walk
step (no eigenvalue above 0.5), while the right-to-left head carries the
rotation. A failure of P5 as written would therefore say nothing against H;
it would only show that the frozen text named the wrong head.

- **P5 stays as written and will still be run,** so the frozen prediction
  gets its honest verdict.
- **P5′ is added below.** It was formulated AFTER seeing the transition
  probe, so it is post hoc and counts as weaker evidence than a frozen
  prediction.

**P5′. The right-to-left head maintains a consistent rotating frame.** In
a hybrid4 that solves SE(2) (seeds 0 and 1 at xs), in the right-to-left
delta-rule head:

- **(a) Consistency.** For step pairs (s, t) with 1 ≤ t − s ≤ 16, the
  product of the head's state updates from step s to step t, in its scan
  order, has an eigenvalue pair at the accumulated heading change between
  the two steps, within ±10 degrees, on at least 95% of pairs. This is
  stronger than the per-step result: it requires every step to rotate the
  same two-dimensional subspace, so that the rotations compose into a
  frame.
- **(b) Decodability.** A ridge probe from that head's state at
  observation s to the pose of step s relative to step t (body-frame dy,
  dx and heading change) reaches R² ≥ 0.8 on held-out episodes, for
  1 ≤ t − s ≤ 16.

**Falsified if (a) fails.** That is, the eigenpair is not at the
accumulated angle on at least 95% of pairs. The clean per-step rotation
would then be local bookkeeping rather than a maintained frame, and the
report must say so.

### 2026-10-15: P5 and P5′ results (seeds 0 and 1)

`tools/probe_p5.py`, hybrid4 SE(2) xs, 512 test episodes (eval seed
4242), on CPU, ridge with an episode-level split.

| check | seed 0 | seed 1 | verdict |
| --- | --- | --- | --- |
| P5 (frozen): left-to-right state at a query token, min R² over lags 1-16 | -0.17 | -0.43 | **fails** |
| P5′(b): right-to-left state at observation s, min R² over lags 1-16 | -0.01 | -0.01 | **fails** |
| P5′(a): rotation pair at the accumulated heading change, spans of 1-16 steps | 99.3% | 98.9% | **passes** |

**P5.** The left-to-right state decodes the pose one step back almost
perfectly (R² 0.98 to 1.00 on every target in both seeds). From two steps
back the heading terms are near zero, and position stays near 0.5. This
fits the contracting, short-memory head the transition probe found.

**P5′(b).** A linear probe of the right-to-left state decodes the pose one
step ahead (position R² 0.76 to 0.81, heading 0.85 to 0.94). Beyond two
steps it is near zero.

**P5′(a).** Across spans of up to 16 steps, the right-to-left head's
combined transition carries its rotation pair at the accumulated heading
change. The same two-dimensional subspace is rotated at every step, so
the per-step rotations compose into a maintained frame. Gaps in seed 1:
100% up to 4 steps, 98.3% at 12, 93.2% at 16.

**Interpretation, post hoc and to be tested.** A delta-rule state is an
associative memory: the relevant stored item depends on which later
anchor is asked about, and the model retrieves it with a content-dependent
query (`S^T q`). A fixed linear readout per lag cannot select that item,
so a failed linear probe does not show the pose is absent. An exploratory
follow-up, not a prediction: a bilinear readout using the model's own
query vectors from each anchor's tokens. The pre-registered P5′(b)
criterion failed, and that verdict stands.

### 2026-10-15: P2 (SE(2) half) — beta capped at 1 fails, and its state does not rotate

hybrid4_beta1 (beta = sigmoid(.) in (0, 1); otherwise hybrid4), SE(2), xs,
per-step queries, 100k steps, seed 0:

- **Result.** All-query mse 0.0491. Bins 0.0137 / 0.0451 / 0.0620 / 0.0723;
  last-step 0.0718; shuffled 0.0918; floor 0.0986.
- **Verdict on the frozen threshold.** At 0.0491 it is past the "failed"
  threshold of 0.045, so P2's SE(2) half is confirmed at this budget.
  hybrid4 reached 0.0180 under the same recipe.
- **Still improving.** The curve was still creeping down at the end, from
  0.0498 at 80k to 0.0491 at 100k.
- **Untested.** P2's Z² half: Z² is on hold.

The mechanism check matches. `tools/transition_probe.py` finds no
eigenvalue pair at 90 or 180 degrees on any turn step: every significant
eigenvalue in the right-to-left head sits at 0 degrees, on every step
class. `tools/read_beta.py` shows the optimiser pressing against the cap.
On the right-to-left head's turn tokens, median beta is 0.99, against a
ceiling of 1.00. On no-turn tokens it is 0.73. So the model drives the
write strength to the largest value it is allowed on exactly the tokens
where hybrid4 uses beta = 2.
