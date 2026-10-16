# se2c build log

Decisions and measurements for the continuous-SE(2) task (`se2c`), kept
by the environment agent. Newest phase last. "User" marks a decision the
user made; everything else is mine and says why.

Machine: local RTX PRO 4000 Blackwell (24 GB), torch 2.14, fla 0.5.2.

## Phase 1: geometry, texture, renderer, coverage, generator, encoding

### Encoding (committed first, d8b0da7)

`mapping/se2c/encoding.py`, `encode_odometry(dxdy, dtheta)` and
`encode_query(dxdy, dtheta)`, both `(..., 50)` float32, the same layout:

- translation: `d / 256`, then sin and cos of `pi 2^k d / 256`, k = 0..9,
  per axis (periods 512 px down to 1 px), 42 features;
- angle: cos and sin of `m theta`, m = 1..4, 8 features.

One scale for both odometry (a few px) and queries (up to ~75 px seen,
encoding unique to +-256 px), so the spec does not depend on the
generator config. **Flag:** the translation argument is in the repo's
`(dy, dx)` order although the interface names it `dxdy`; the spec said
"(dx, dy, dθ)" for odometry and "(dy, dx)" for queries. I used (dy, dx)
for both, as everywhere in `mapping/`; the features treat both axes alike,
so only consistency matters. The blind-probe agent should use (dy, dx).

### Geometry (`mapping/se2c/geometry.py`)

Heading 0 faces up (-y), positive turns are right turns (clockwise on
screen); facing direction `(-cos θ, sin θ)`. At quarter turns `rotate`,
`unrotate` and view sampling equal `mapping.se2` exactly (tested,
including `se2.face` of a crop). A view samples the world at
`p + unrotate((i - 7, j - 7), θ)`; its footprint is the convex hull of
its 225 sample points (half-width 7, not 7.5), so every covered target
point can be interpolated from one observation without extrapolation.
Steps integrate along the arc, `arc(v, ω) = v (-sin ω/ω, (1 - cos ω)/ω)`
(checked against numerical integration). All in fp32.

### Walk (user: isotropic arc walk, ω ~ U[-π, π), granularity, decoupled mode, held-out bands)

- **v ~ U[3.75, 7.5] px = [S/4, S/2]** (mine). Reasons: consecutive views
  always overlap by at least half a window (keeps visual odometry
  possible later); the agent always moves; query offsets stay well
  inside the encoding's ±256 px. Measured (8 eval batches): walk bounding
  box median 42 px, net displacement median 33 px, query offset median
  10.8 px, p90 29.8, max 76.
- **Concern (flag):** with ω_max = π the walk has no persistence: the
  agent dithers inside a ~40-px area, and the map an episode covers is
  smaller than the discrete task's (decoupled grid mode: bbox 72 px,
  offsets median 21.5 px). A smaller ω_max gives a more robot-like path
  but breaks query sampling (next section): at ω_max = π/4 the
  footprints form a 15-px corridor in which a view at a free heading
  never fits, and 98.5% of queries fall back.
- Granularity `turn_n` in {4, 8, 16, 32} or 0 (continuous): turns, the
  first heading and query heading changes lie on the lattice 2πk/n, k in
  [-n/2, n/2); angles are snapped to the lattice after composition so
  fp32 drift cannot turn -π into +π (a real bug the tests caught).
- Decoupled mode: world move directions on the lattice with no
  immediate reversal, headings i.i.d. With `turn_n=4, v=9` the 16
  odometry symbols match the discrete SE(2) generator's frequencies
  (total variation < 0.03), the moves are exact ±9-px grid steps, and
  `se2.compose` of the recovered symbols reproduces the poses (test).
- Held-out band on |ω| (`holdout=(lo, hi)`, `holdout_mode` exclude /
  include / only), continuous or lattice. Applies to step turns only;
  query heading changes are not restricted (flag: a choice, the spec
  named ω only).

### Texture (user: procedural, band-limited, isotropic, exact; tune and LOOK)

z = sqrt(2/K) Σ cos(2π κ_k·u + φ_k), K = 96, directions and phases
uniform, |κ| in [f_lo, f_hi] with density ∝ f^-slope; unit variance.
Rendered exactly at the rotated pixel centres; a view is one separable
(15 x K)(K x 15) product (agrees with the direct sum to 2e-4 and with
the float64 NumPy reference to 5e-4). Deterministic from the seed.

Tuning (`/tmp/.../lift_env/` scripts; numbers on 16 episodes, 128 px,
4x supersampled). "alias" = power of g above 0.5 cyc/px; "bicubic" =
error of bicubic interpolation from the unit grid at random off-grid
points, as a fraction of g's variance (what the pixels cannot pin down):

| variant | alias | bicubic | look |
| --- | --- | --- | --- |
| linear (no nonlinearity), 1/f, [1/48, .2] | 0.003 | 0.0004 | grey blur, hopeless to eyeball |
| sigmoid c3, 1/f, [1/48, .2] | 0.004 | 0.0015 | soft blobs |
| sigmoid c6, 1/f, [1/48, .2] | 0.010 | 0.008 | crisp blobs, generic shapes |
| sigmoid c4, 1/f, [1/32, .18] | 0.006 | 0.003 | crisp blobs; 2 px / 10° errors hard to see |
| stroke w.35, 1/f, [1/48, .2] | 0.049 | 0.052 | strokes of mixed width, aliased |
| stroke w.3, flat, [.05, .15] | 0.097 | 0.109 | thin strokes, badly aliased |
| stroke w.3, flat, [.025, .08] | 0.011 | 0.009 | MNIST-like strokes |
| **stroke w.35, flat, [.025, .08]** | 0.005 | **0.0045** | **chosen** |

**Chosen:** `nonlinearity='stroke'`, g = exp(-z²/(2·0.35²)): white lines
along the zero set of z, about 3 px wide, 12-40 px apart, on black (mean
0.33). Spectrum flat in radius (slope 0) over [0.025, 0.08] cyc/px,
about 1.7 octaves. Why: (1) it looks like pen strokes, so a correct
prediction, a shifted one and a blurred one are different pictures
(mockup.png); blobs from the 1/f + sigmoid family made 2-px/10° errors
hard to see; (2) a narrow band gives lines of nearly constant width,
whereas a 1/f band gives fine-scale zero crossings whose lines are
sub-pixel and alias (5%); (3) the residual not pinned down by the pixels
is 0.45% of the variance. **Deviation (flag):** the user suggested a
roughly 1/f spectrum; I chose a near band-pass one for the reasons
above. Blank views are rare: 0% of views have under 5% ink, 0.5% under
10%. The other families stay available through `TextureSpec`.

Sheets (`python tools/se2c_sheets.py`, docs/lift/sheets/):

- `texture.png`: world patches of four episodes; views of one place at
  0, 20, 45, 90, 135, 200 degrees.
- `mockup.png`: targets beside 0.5 px/3°, 1 px/5°, 2 px/10°, 4 px/30°
  pose errors, Gaussian blurs σ = 1, 2, 3, and the floor, with MSE.
- `episode.png`: an episode on its world texture, observation footprints
  (yellow), the walk (red), query footprints (green found, magenta
  fallback), the observations and targets.
- `bank.png`: the MNIST bank option.

What the mockup says about MSE: per view, 0.5 px/3° costs ≈ 0.02, 1 px/5°
≈ 0.06, 2 px/10° ≈ 0.14, which is the floor. Sharp strokes make MSE
punish small pose errors as hard as drawing nothing, and a blur σ = 1
(≈ 0.013) scores better than a 0.5-px error. Expect a model under pose
uncertainty to blur; the grid will show it.

Tile banks: `Config(bank='mnist' | 'imagenette' | 'imagenette_hp')`
builds a random dihedral tile map on the GPU and renders by bilinear
interpolation, supersampled 4x4 per pixel over the pixel's rotated square
(box filter). With aa = 1 at quarter turns it equals the discrete crop
(test).

### Queries and coverage (user: causal coverage is the contract)

- **Raster:** first-seen step per 0.25-px cell, a cell counting as seen
  only if it lies wholly inside a footprint (so the test is never
  optimistic; held against the exact reference on random points, never
  violated). It loses 2.5% of the truly covered area at the edges (5.0%
  at 0.5-px cells).
- **Sampler:** centre uniform over the union of footprints of steps <= t
  (draw a footprint, a point in it, keep with prob 1/multiplicity),
  heading change uniform; stage 1 tests centre, corners and edge
  midpoints for 512 candidates, stage 2 tests all 225 points for the
  first 16 survivors; the first passing candidate is taken, which is
  uniform over the valid poses (up to the raster margin).
- **Exact check:** every target point of every query lies in the
  footprints of steps <= t (causal) or <= T (global), against the
  float64 reference, for the default walk, n = 8, decoupled, slow walks
  and global coverage. Global queries do violate causal coverage (so the
  variants differ).
- **Fallback:** if no candidate passes, the query is the exact pose of a
  random observation s <= t turned by a random multiple of 90° (covered
  exactly). Rate 17% overall at the defaults, almost all at t <= 5
  (t = 0-3: ~100%, t = 4: ~50%, t = 8: ~13%, t >= 12: < 5%): early
  footprints leave no room for a view at a free heading. In the
  discrete task only t = 0 was forced. **Flag:** about a sixth of the
  queries are "re-render observation s at a quarter turn", which still
  needs the pose of s relative to t but not interpolation.
- **Cost:** a whole batch of 64 episodes takes 16.7 ms wall (about 9 ms
  of GPU time); the raster is 6.7 ms of it, the query sampler ~10 ms,
  rendering 1 ms. **Causal coverage is affordable, so it is the
  default**; `query_coverage='global'` exists as the documented
  deviation and is not used.

### Oracle ceiling and floor

`mapping/se2c/oracle.py`, true poses given:

- `analytic` (render from the episode's own function): 0.0 exactly, a
  consistency check only.
- `interpolate`: each target point from the observation (<= t) it lies
  deepest in. Bilinear 0.0017, **bicubic 0.0005** MSE (eval batches).
  Against a constant floor of ≈ 0.13 that is 0.4% of the floor: the task
  imposes almost no residual; the bicubic number is the reported
  ceiling.
- A kriging oracle (best linear predictor from all observed pixels under
  the texture's empirical covariance) was tried and dropped: the
  tabulated covariance is not positive definite and the noise-free
  system was unstable (worse than bilinear).

## Phase 2: trainer and harness (commit e68c17f)

Shared files touched, each minimally:

- `mapping/tasks.py`: a `GPU_TASKS = {'se2c': ...}` table beside `TASKS`
  (not inside it: several existing tests iterate `TASKS` assuming the
  tile-bank tasks), `ALL`, and `get()` / `config_and_spec()` aware of it.
- `train.py`: `--task` choices from `tasks.ALL`; the se2c generator
  flags (`mapping.task_se2c.add_args`); a `gpu_data` branch that takes
  config, spec, the batch stream, the held-out set and run info from
  `mapping.task_se2c.data` instead of the bank loader (the old code is
  unchanged, only indented under `if not gpu_data`); `permute_steps` so
  the shuffled-odometry diagnostic permutes (B, T, 3) pose vectors (for
  (B, T) symbols it is the old gather, tested); an optional per-query
  breakdown hook (`metrics_by_query`); `offset_max` read with getattr.
- `render.py`: se2c held-out batches from `task_se2c.data`.
- `mapping/render.py`: the same permutation fix; float query poses are
  labelled in px and degrees.
- `README.md`: layout, commands, docs entry.

The z2 and se2 batch hashes and every pre-existing test pass unchanged
(209 tests at this commit, with the blind-probe and ablation agents'
tests). Evaluation: 16 fixed batches from the evaluation seed stream;
with a held-out band the evaluation stream includes it. Floor on the
default eval set (16 batches, seed 0): mse 0.1311, c-mse 0.1192, fg_mae 0.4828. Copying the observation at the query step instead: 0.248. Pose oracle (bicubic) by prefix bin: 0.00034 / 0.00054 / 0.00058 / 0.00057; by |turn| bin 0.00045-0.00055.

## Phase 3: model ports

`mapping/models/se2c/hybrid4c.py` and `vanilla4c.py`; mixers, blocks,
ladder, LR and init are imported from the se2 `hybrid4` / `vanilla4`
modules so they cannot drift. The mechanism trace (how the turn reaches
the delta rule's key and beta, and why a continuous key must have a cut
somewhere in theta) is in hybrid4c's docstring.

Parameters (3 x 3 patches):

| rung | hybrid4c | vanilla4c |
| --- | --- | --- |
| xs | 124,717 | 119,497 |
| s | 870,617 | 832,393 |
| m | 2,829,069 | 2,703,945 |
| l | 6,788,553 | 6,496,521 |
| xl | 22,336,473 | 21,366,153 |
| xxl | 53,082,121 | 50,800,137 |

(hybrid4 xs on se2: 119,661; the difference is the 25-entry patch and
query tables and the 50-feature linear embeddings.)

Tests (`tests/test_se2c_models.py`): shape, episode isolation, None equal
to all-last-step, patchify layout and round trip, placement (all queries
at the last step = the original layout; a query is contiguous, right
after observation t's last patch, before odometry token t; two queries
at one step keep query order), the RoPE step every token carries, and
that the odometry enters only through the shared, 2π-periodic encoding.

Throughput, batch 64, bf16, steps/s (forward, backward, AdamW). FIRST
MEASUREMENT, taken while the blind-probe agent trained on the same GPU
outside the lock, so these are lower bounds; remeasured below before the
proving run:

| model, rung | 3x3 fixed batch | 3x3 with generation | 5x5 fixed | 5x5 with generation |
| --- | --- | --- | --- | --- |
| hybrid4c xs | 39.1 | 39.9 | 82.7 | 38.1 |
| hybrid4c s | 9.0 | 8.5 | 25.3 | 18.4 |
| hybrid4c m | 3.6 | 3.7 | 13.8 | 9.1 |
| vanilla4c xs | 55.8 | 31.0 | 127.3 | 44.7 |
| vanilla4c s | 14.0 | 11.6 | 42.2 | 27.1 |
| vanilla4c m | 5.7 | 6.1 | 18.8 | 15.0 |

Generator alone: 14 ms per batch of 64.

**Clean remeasurement** (under the GPU lock; the blind probe's own run
was not under the lock, so a small residual contention is possible):

| model, rung | 3x3 fixed batch | 3x3 with generation | 5x5 fixed | 5x5 with generation |
| --- | --- | --- | --- | --- |
| hybrid4c xs | 71.6 | 46.9 | 167.3 | 77.3 |
| hybrid4c s | 16.3 | 14.6 | 48.7 | 36.2 |
| vanilla4c xs | 108.3 | 61.8 | 247.9 | 92.0 |
| vanilla4c s | 27.2 | 22.8 | 81.8 | 51.9 |

Generator alone 6.8 ms per batch of 64; peak memory at s (3x3): hybrid4c
2.5 GiB, vanilla4c 1.8 GiB. Readings:

- 3x3 patches cost about 2.3-3x the 5x5 tokeniser's compute at a fixed
  batch (1057 against 401 tokens). At xs it does not matter (a 25k run
  is 10 minutes). At s, 100k steps is about 2 h locally; m is about 3x
  slower again. **Not switched.** Proposal for the scaled sweeps only:
  keep 3x3 at xs/s locally, and measure on the H200 before deciding;
  if 5x5 is wanted, it is one constructor argument (`patch=5`) and a
  separate model name, so the 3x3 curves stay comparable.
- At xs the generator, not the model, limits the step (71.6 -> 46.9):
  it is CPU launch-bound (~150 small kernels). Prefetching the next
  batch on a side CUDA stream or a CUDA graph would recover most of it;
  not done yet.

## Phase 4: smoke and the proving run

Smoke, 300 steps (outside the lock, as allowed): runs end to end, eval
0.1237 against floor 0.1311, shuffled equal (as expected so early).

Proving run, under the shared GPU lock: `python train.py --task se2c
--model hybrid4c --size xs --steps 25000 --query-times uniform
--eval-every 500 --log-every 100 --run seed0_25k_qt` (cosine schedule,
lr 3e-4, batch 64), 45.5 it/s, 9.5 minutes. Run directory
`results/se2c/hybrid4c/xs/seed0_25k_qt/` (not in git; grid and curve
copied to `docs/lift/sheets/proving_hybrid4c_xs_25k_{grid,curve}.png`).

| step | eval mse | shuffled | last-step |
| --- | --- | --- | --- |
| 2k | 0.0911 | 0.0912 | 0.1260 |
| 4k | 0.0759 | 0.0759 | 0.1085 |
| 10k | 0.0710 | 0.0713 | 0.1042 |
| 16k | 0.0683 | 0.0705 | 0.1025 |
| 20k | 0.0665 | 0.0715 | 0.1011 |
| 25k | **0.0655** | **0.0715** | 0.1006 |

Final by prefix bin 0.029 / 0.062 / 0.077 / 0.090; by |heading change|
0.066 / 0.062 / 0.069 / 0.066 (flat). Floor 0.1311, oracle 0.0005.

Reading: the pipeline works; the model is not solving. It learns the
easy part first: early queries (where a sixth of queries are fallback
re-renders of an observation, and targets overlap the anchor view most)
and blurred local content; the odometry gap opens only after ~14k steps
(0.0655 against 0.0715 shuffled) and is still widening, and the
last-step error is near the floor. For scale, the discrete hybrid4 xs on
MNIST at the same 25k steps was 0.0543 against shuffled 0.0928 (floor
0.0986) and needed about 90k steps to meet "solved"; so this is
consistent with an xs model early in its curve, not evidence either
way. The grid: the good rows (mse ~0.01) are re-renders near the anchor
or fallbacks, equally good with shuffled odometry; the long-range rows
are grey blur.

Checks done after the run, on 8 evaluation batches:

- Non-fallback targets are not copies of any observation: the best
  observation, at the best quarter turn, matches within MSE 0.02 for
  only 1% of them (18% of all queries, i.e. the fallbacks).
- Query centres lie a median 2.0 px (p90 3.6 px) from some observation
  centre up to t: the free-heading coverage constraint pins queries to
  where the agent has been; the headings are free, so the task is a
  rotation and resampling, not a lookup.
- On average 22% of a (non-fallback) target's points lie in the anchor
  observation's own footprint (44% in the first prefix bin, 14% in the
  last): most of each target must come from other steps.
