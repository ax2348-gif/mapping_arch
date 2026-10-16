# Model contract for the SE(2) mapping task

This is the second rung of the study. It is a **separate task** from the
Z² task in `CONTRACT.md`, with its own batch layout, its own model
interface, and its own results tree. Read `CONTRACT.md` first: everything
there about why the study exists, building as an ML architect, and what
counts as cheating carries over unchanged. This file states only what is
different.

## The task

The same map, the same random walk of 32 steps of 9 pixels sampled in the
world frame, the same 9x9 windows. In addition the agent has a **heading**
at every step, drawn uniformly and independently from the four cardinal
directions. Every observation is presented in the agent's own frame,
rotated so that its heading is image-up. The agent does not know the
world frame; it knows only how each pose relates to the previous one.

| input | shape | meaning |
| --- | --- | --- |
| `obs` | `(B, 33, 9, 9)` float in [0, 1] | the observations, each facing its own heading |
| `odometry` | `(B, 32)` long in 0..15 | relative pose from t to t+1 in the body frame of t: `4 * turn + move`, with `move` in {forward, right, back, left} and `turn` in {none, right, around, left}; `mapping.se2.decode` splits it |
| `query_pose` | `(B, 8, 3)` long | the target window relative to the pose at the query's step: body-frame `(dy, dx)` within ±288, and the heading change in 0..3 |
| `query_time` | `(B, 8)` long in 0..32 | the step each query is asked at, as in `CONTRACT.md`; all 32 by default |

Output: `pred (B, 8, 9, 9)` float, the window at each query **facing the
query heading**. Loss is MSE. As before, every target pixel was observed,
queries fall at arbitrary pixel offsets, and the setting is offline with
full attention and relative odometry only.

`mapping.se2` fixes the conventions and `tests/test_se2.py` checks them;
`python -m pytest` must pass before any model is trained.

## Generator settings: step size, odometry dropout, texture

The three generator settings of `CONTRACT.md` ("Generator settings") apply
here unchanged and do not make a new task. `--step-px` sets the walk step
(`TaskSpec.step_px`, `offset_max = steps * step_px`); below 9 consecutive
observations overlap. `--odometry-dropout p` replaces each odometry symbol,
with probability p, by the reserved **unknown** symbol 16; when p > 0
`TaskSpec.odometry` is 17 so tables sized from it get the extra row, and a
model must not decode 16 as a (turn, move) pair. The true symbols ride in
the batch as `odometry_true`, in `FORBIDDEN_KEYS` beside `positions` and
`headings`. With p = 1 and a step below 9, the relative pose between
frames (move and turn) is recoverable only by registering the overlap of
consecutive observations; `tools/registrability.py` measures how often
that is possible at all on MNIST maps, which is the ceiling for any model.

`--texture imagenette` builds the map from 64 x 64 grayscale natural-image
tiles instead of MNIST digits (no brightness cutoff, start window one
tile). It exists for the visual-odometry setting: MNIST overlaps are
often blank and so cannot be registered, natural ones almost never are
(`tools/registrability.py --texture imagenette`). Observations stay 9 x 9
single channel and face their heading exactly as before; dihedral tile
augmentation stays on, so the upright orientation of natural content
cannot reveal the heading. The interface, the spec and the rules do not
change.

## Why this rung

In the Z² task, placing a frame means summing the moves: one abelian group
operation, and a linear recurrence with a constant decay does it. Here
placing a frame means composing poses in the semidirect product Z4 ⋉ Z²:
each body-frame move must be rotated by the heading accumulated up to that
step before it is summed, and the rotation itself is a running total of
turns. The composition is non-commutative, so a prefix sum over the
odometry tokens is insufficient. On top of that, the fragments arrive in
four orientations and must be aligned before they can be assembled. The
hybrid arm solved the Z² task at a tenth of the parameters of the general
transformer; this rung asks whether that bias was integration in general
or summation in particular.

## What is different in the rules

All exclusions in `CONTRACT.md` apply, read with "odometry" for
"directions" and "query pose" for "query offset". In addition:

- **No external composition of poses.** No cumulative sum of turns, no
  rotation of moves by a computed heading, no `mapping.se2.compose` or any
  re-implementation of it outside learned layers. A layer that learns a
  state which rotates is the point; a hand-written one is not.
- **No un-rotating the observations.** The observations arrive facing
  their heading and must be consumed as given. Rotating them by anything
  derived from the odometry is external geometry.
- **No use of `positions`, `headings` or `odometry_true`.** They are in
  the batch for diagnostics only.
- **Query pose is a raw input.** Embed the three integers however you
  like; do not combine them with the odometry by arithmetic the model did
  not learn.

## Files and interface

- One module `mapping/models/<name>.py` as before. `forward(obs, odometry,
  query_pose, query_time=None)` returns `pred`; the query-times rules of
  `CONTRACT.md` apply. The task spec is `mapping.task_se2.TaskSpec`
  with `view`, `steps`, `odometry` (16), `turns` (4), `step_px`,
  `queries`, `offset_max`.
- A Z² model is **not** expected to run on this task. Porting one is its
  own contribution with its own file, named `<name>_se2`, and its own doc.
  The port should change the interface and nothing else where possible,
  so the two curves are comparable; say in the doc exactly what changed.
- Results live under `results/se2/<model>/<size>/<run>/`, and
  `summarize.py --root results/se2` reads them, so the two tasks never
  share a table or a plot by accident.

## Deliverables

As in `CONTRACT.md`: the model file with its audit section, a doc page,
the smoke run against the constant floor, the ladder printout, and the
largest-rung memory check. The shuffled diagnostic permutes the odometry
symbols in time; report it beside the plain number as before. The grid
shows observations and targets in their own frames, so expect rotated
fragments in it.

## Status

Generator, `mapping.se2`, tests, the task spec, the trainer's `--task
se2` switch and the hybrid port (`mapping/models/se2/hybrid.py`) exist.
No other model is ported yet.
