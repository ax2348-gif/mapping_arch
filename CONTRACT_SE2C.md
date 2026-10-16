# Model contract for the continuous SE(2) task (`se2c`)

The third rung, a **separate task** from Z² (`CONTRACT.md`) and discrete
SE(2) (`CONTRACT_SE2.md`), with its own generator, batch layout, model
registry (`mapping/models/se2c/`) and results tree (`results/se2c/`).
Read both earlier contracts first: why the study exists, building as an
ML architect, and what counts as cheating all carry over. This file says
what is different. Design notes and measurements: `docs/lift/`.

## The task

The agent's pose is continuous: a position (y, x) in pixels and a heading
θ in radians. Each of 32 steps drives forward at a speed v ~ U[3.75, 7.5]
px while turning by ω ~ U[-π, π), along the arc (the SE(2) exponential).
At each of the 33 poses it sees a 15 x 15 view rotated so that its
heading is image-up, rendered EXACTLY from a procedural texture (white
strokes on black, drawn fresh per episode; `mapping.se2c.texture`). It is
then asked to render the 15 x 15 view at a query pose.

| input | shape | meaning |
| --- | --- | --- |
| `obs` | `(B, 33, 15, 15)` float in [0, 1] | observations, each facing its heading |
| `odometry` | `(B, 32, 3)` float | pose t+1 seen from pose t: body-frame (dy, dx) px (forward is -dy, right is +dx) and the turn in [-π, π), positive = right |
| `query_pose` | `(B, 8, 3)` float | the query view's centre seen from the pose at the query's step: body (dy, dx) px, and the heading change |
| `query_time` | `(B, 8)` long in 0..32 | the step each query is asked at |

Output `pred (B, 8, 15, 15)`, the view at each query facing its heading.
Loss is MSE. `mapping.se2c.geometry` fixes the conventions; at quarter
turns they are exactly `mapping.se2`'s (`tests/test_se2c_geometry.py`).

**Coverage (the contract on queries).** Every target sample point of a
query asked at step t lies inside the union of the footprints (rotated
squares: the convex hull of a view's sample points) of observations
0..t. Checked point by point against an exact reference
(`tests/test_se2c_coverage.py`). The query pose is relative to the pose
at t; a model places the query's tokens after observation t and gives
them step t. Query poses are uniform over the valid poses; when the
sampler finds none (mostly t <= 5: early footprints leave no room for a
view at a free heading), the query is the exact pose of a random
observation s <= t turned by a multiple of 90°, which is covered
exactly (about 17% of queries; `docs/lift/build_log.md`).

`query_coverage = "global"` (queries anywhere covered by the whole
sequence, still anchored and placed at their own step t) exists as a
**deviation** and is not the default; causal coverage costs about 10 ms
per batch of 64 and is used.

## Inputs are raw; the encoding is shared

The pose inputs are raw floats. Models embed them with the shared,
parameter-free feature map `mapping.se2c.encoding` (`encode_odometry`,
`encode_query`: scaled scalars and octave Fourier features of the
translation, cos and sin of m θ for m = 1..4) followed by learned layers.
That is an embedding of a raw input, allowed by `CONTRACT.md`. All the
earlier rules apply, read with "odometry" for "directions":

- No composition of poses outside learned layers: no cumulative sum of
  turns or moves, no rotation of a move by a computed heading, no use of
  `mapping.se2c.geometry.integrate` or a re-implementation.
- No un-rotating observations by anything derived from the odometry.
- RoPE and placement on the sequence index (step) only.
- `poses`, `query_world`, `texture_params`, `stats` and `raster` are in
  the batch for diagnostics only (`mapping.task_se2c.FORBIDDEN_KEYS`).

## Generator settings (not new tasks)

`train.py --task se2c` takes (`mapping.task_se2c.add_args`):

- `--turn-n n` (n in 4, 8, 16, 32; default 0, continuous): turns, the
  first heading and query heading changes on the lattice 2πk/n. The
  granularity curriculum.
- `--motion decoupled`: heading drawn independently of the motion, world
  moves on the lattice with no immediate reversal, as in the discrete
  task; with `--turn-n 4 --v-min 9 --v-max 9` the odometry statistics
  are the discrete SE(2) task's (tested).
- `--holdout LO HI` with `--holdout-mode exclude|include|only`: a band of
  |ω| left out of training; evaluation includes it.
- `--v-min`, `--v-max`, `--omega-max`; `--query-coverage causal|global`.
- `--se2c-texture procedural|mnist|imagenette|imagenette_hp`: the tile
  banks as an option, rendered with bilinear interpolation and 4 x 4
  supersampling per pixel (antialiasing).
- `--query-times uniform|last` as before (`uniform` for every run so far).

Batches are generated on the GPU inside the training step
(`mapping.se2c.generator`, deterministic from the seed and batch index);
evaluation uses a fixed set from a separate seed stream (with the
held-out band included, and the test split of a bank).

## Metrics

As before (`mse`, `mse_centered`, per prefix-length bins, last-step, the
shuffled-odometry diagnostic which permutes the odometry vectors in
time), with foreground = ink (target > 0.5), plus `mse_turnA_B`: error
per |query heading change| bin. References on the default settings: the
constant floor is 0.131 (16 evaluation batches, seed 0); a pose-oracle that interpolates the
observations bicubically with the TRUE poses reaches 0.0005
(`mapping.se2c.oracle`). Proposed reading (not frozen):
**solved ≤ 0.02** (about a 0.5 px / 3° pose error, or a 1-px blur, per
view; `docs/lift/sheets/mockup.png`), **failed ≥ 0.065** (half the
floor). The per-query MSE of a 2-px / 10° pose error is already at the
floor: on sharp strokes MSE does not grade near misses, so look at the
grid.

## Files and interface

One module `mapping/models/se2c/<name>.py` with `SIZES`, `SCALING_RULE`,
optional `LR`, `build(size, spec)` and `register('<name>')`; `forward(obs,
odometry, query_pose, query_time=None)`; `spec` is
`mapping.task_se2c.TaskSpec` (`view`, `steps`, `queries`,
`odometry_features`, `query_features`). A port of an SE(2) model is named
`<name>c` and changes the interface and nothing else where possible
(`hybrid4c`, `vanilla4c`).

## Status

Generator, coverage, oracle, encoding, the harness path and the ports
`hybrid4c` and `vanilla4c` exist. First run: hybrid4c xs, 25k steps,
eval 0.0655 (shuffled 0.0715, floor 0.131): the pipeline is proven, the
task is not yet solved at that budget (`docs/lift/build_log.md`).
