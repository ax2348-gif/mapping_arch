# Model contract for the continuous SE(2) task (`se2c`)

The third rung, a **separate task** from Z² (`CONTRACT.md`) and discrete
SE(2) (`CONTRACT_SE2D.md`), with its own generator, batch layout, model
registry (`mapping/models/se2c/`) and results tree (`results/se2c/`).
Read both earlier contracts first: why the study exists, building as an
ML architect, and what counts as cheating all carry over. This file says
what is different. Design notes and measurements: `docs/lift/`.

## The task

The agent's pose is continuous: a position (y, x) in pixels and a heading
θ in radians. Each of 32 steps, as in the discrete task, MOVES and then
TURNS, independently (`motion='decoupled'`, the default since 2026-10-16
by the user's decision):

- **Move:** v ~ U[4.5, 9] px in a world direction that changes by
  δ ~ U[-3π/4, 3π/4) per step (`directions='persistent'`). This is the
  continuous form of "no immediate reversal": E cos δ = 0.30 against
  the discrete walk's 1/3, and it is the same at every turn granularity
  n, so the walk's spread is constant along the curriculum.
- **Turn:** ω i.i.d., uniform on [-π, π) or on the n-lattice, so the
  headings are i.i.d. uniform as in the discrete task.

The first walk, driving along an arc while turning by ω ~ U[-π, π) at
v ~ U[3.75, 7.5], stayed in a ~40 px blob (queries a median 10 px from
their anchor; `docs/lift/sheets/episode_arc.png`). It remains available
as `--motion arc`. The decoupled walk ends a median 40 px from its start,
with queries a median 15 px from their anchor (`episode_decoupled.png`).

**The odometry is the relative pose, in an egocentric parameterisation.**
The walk is a sequence of poses g_0 ... g_32 in SE(2); nothing happens
between them. There is no path, so no parameterisation of a step is more
"physical" than another. What the model needs is the relative pose
g_t^-1 g_(t+1), the next pose seen from the current one. Any egocentric
parameterisation of it carries the same information and is fair game:

- move-then-turn, (body-frame translation in pose t's frame, turn);
- turn-then-move, the translation in pose t+1's frame;
- the twist, log(g_t^-1 g_(t+1)), the constant-velocity arc between the two poses.

None of these is allowed to be allocentric: no world-frame direction, no
absolute heading, nothing integrated across steps (the earlier rules).

**This task uses move-then-turn** (`mapping.se2c.geometry.odometry`): the
body-frame (dy, dx) of pose t+1 in pose t's frame, then the turn wrapped
to the minimal angle in [-π, π). A half turn is -π; on the n-lattice the
turn is snapped to 2πk/n exactly. This is the discrete contract's
convention, and its parts are plausibly the easiest to predict from:
the translation reads off pose t's view directly. The generator also
DRAWS the walk this way (move, then turn). That is a sampling choice
about which pose sequences occur, not a claim about motion between poses.

The shared encoding feeds the model cos and sin of mθ, so the choice of
angle representative changes nothing a model sees. It matters only to
code that reads the raw angle: held-out bands, probes and metrics. A
different parameterisation, such as the twist, would be a task variant:
the same episodes and targets, a different odometry input, documented
here before use.

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
turns they are exactly `mapping.se2d`'s (`tests/test_se2c_geometry.py`).

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

## Turn granularity n (the curriculum)

`--turn-n n` puts every ANGLE in the episode on the lattice 2πk/n and
leaves every TRANSLATION continuous. n = 0 (the default) is the continuous
task.

| on the lattice when n > 0 | not on the lattice |
| --- | --- |
| every turn ω (so every heading, relative to the first) | move lengths v ~ U[4.5, 9] |
| the first heading | move directions (`persistent`: continuous at every n) |
| each query's heading change from its anchor | query centres (anywhere covered) |

- **The group is C_n ⋉ R².** Only the rotation part is discretised. Even
  at n = 4 the default task is not the discrete task: the translations
  are continuous, so sub-pixel interpolation is needed at every n. The
  **bridge** (`--directions lattice --turn-n 4 --v-min 9 --v-max 9`) also
  puts the translations on the 9 px grid. It is the discrete task's walk
  with the new renderer, texture and encoding.
- **The walk does not change with n.** For the same seed, the positions
  are identical at every n; the move directions are drawn before, and
  separately from, the turns. So the curriculum varies the rotation group
  alone.
- **What changes for the model.**
  - At n = 4, a view and its quarter-turned twin share a pixel grid, so
    rotation is a pixel permutation.
  - At n >= 8, rotated views relate by resampling. Rendering is analytic,
    so no interpolation artefacts are in the data.
  - The angle encoding (cos and sin of mθ, m = 1..4) separates every
    lattice angle at every n; at n = 4, m = 4 is constant.
- **Why it is the lever for the seam.** A turn by θ, made exactly as a
  product of delta-rule reflections, needs keys whose pair is set by θ/2.
  - On a finite lattice, the n required key settings are isolated
    inputs, so a network can fit them exactly at any n.
  - On the continuum, keys continuous in θ cannot do it over the whole
    circle: the full turn is the nontrivial loop of SO(d), and it does
    not lift to a closed loop of reflection products (docs/STATUS.md,
    session log).
  - Between those limits, the key map must flip sign across one gap of
    width 2π/n, so its steepness grows with n.
  - Expected, not yet frozen: the learning breakaway comes later as n
    grows, and the error concentrates in one band of turn angles (the
    seam), visible in `mse_turnA_B` and in the transition probe.
- **Fallback queries.** These re-render an observation at a quarter turn,
  which is on every lattice in the curriculum (n = 4, 8, 16, 32).
- **Held-out bands** (`--holdout`) are bands of |ω|. With n > 0 they
  remove lattice angles, so a band must contain at least one of them.

## Generator settings (not new tasks)

`train.py --task se2c` takes (`mapping.task_se2c.add_args`):

- `--turn-n n` (n in 4, 8, 16, 32; default 0, continuous): the
  granularity curriculum, above.
- `--motion decoupled|arc` (default decoupled) and `--directions
  persistent|lattice` (default persistent). `lattice` takes move
  directions on the n-lattice with no immediate reversal; with `--turn-n
  4 --v-min 9 --v-max 9` the odometry statistics are the discrete SE(2)
  task's (tested). This is the bridge setting.
- `--holdout LO HI` with `--holdout-mode exclude|include|only`: a band of
  |ω| (the turn) left out of training, in either motion; evaluation includes it.
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
per |query heading change| bin. Also, per query kind (added 2026-10-16 at the user's request):
`mse_fallback` (an earlier observation at a quarter turn, the sampler's
fallback), and the sampled queries by their distance from the anchor:
`mse_near` (< 7.5 px, a re-render of the anchor view), `mse_mid`
(7.5-15 px) and `mse_far` (>= 15 px, beyond a window, so it needs the
map), plus `mse_sampled` and each kind's share `frac_<kind>`. The grid
groups its rows by the same kinds. Read progress on `mse_far`, not on
the aggregate. References on the default settings: the
constant floor is 0.1307 on the decoupled default (0.1311 on the old arc walk; 16 evaluation batches, seed 0); a pose-oracle that interpolates the
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
