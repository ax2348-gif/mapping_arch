# The continuous SE(2) lift: plan and decisions

Reference: the user's lift notes, https://claude.ai/artifact/LUSCC1MToip91haQkC28zc
("Continuous Lift & hybrid4.5", a pre-freeze draft: L1-L5, hybrid4.5b, §06
checklist). One correction to them: hybrid4_noconv HAS a result. It failed at
0.0537 against 0.0180; see HYPOTHESIS.md.

## Goal

Lift the mapping result from discrete Z4 ⋉ Z² to continuous SE(2), as the
fastest route to robotics. The rung's job is to show that the operator,
ported properly, works with continuous turn angles and continuous view
rotation. The user's prior is that a properly ported hybrid4 will likely just
work. Iterate locally first.

## Decisions settled with the user (2026-10-16)

- **Task.** A new task, `se2c`, organised as its own unit:
  - `mapping/se2c/`: geometry, texture, rendering, coverage, generator, `encoding.py`
  - `mapping/task_se2c.py` and a `se2c` entry in `mapping/tasks.py`
  - `mapping/models/se2c/`: `hybrid4c`, `vanilla4c`
  - tests in `tests/test_se2c_*.py`
  - results in `results/se2c/`
  - the contract in `CONTRACT_SE2C.md`
  - docs here, in `docs/lift/` (`build_log.md` is kept by the environment agent)

  z2 and se2 stay byte-identical; the golden tests must pass.
- **Trajectory: an isotropic walk.** Each step draws a forward speed v from
  U[v_min, v_max] with v_min > 0, and a turn ω from U[-ω_max, ω_max), half-open,
  with ω_max = π by default. The step is integrated along the arc with the SE(2)
  exponential. Options:
  - **Granularity:** ω quantised to 2π/n for n in {4, 8, 16, 32}, or continuous. This is the L1 curriculum.
  - **Decoupled mode:** heading independent of motion, matching the discrete statistics; n = 4 should reproduce them.
  - **Held-out angle bands:** some ω excluded in training, for the eigenangle calibration.
- **Observation:** 15x15, up from 9x9, to limit aliasing. Models use 3x3
  patches by default (25 tokens per view). Switch to 5x5 only if 3x3 proves
  too slow, and only with the numbers reported.
- **Texture: procedural, band-limited and isotropic.**
  - **Form.** A random field, for example a sum of random cosines with a roughly 1/f spectrum below Nyquist, drawn fresh per episode.
  - **Rendering is exact.** The field is evaluated at the rotated pixel centres.
  - **Generation runs on the GPU inside the training step,** with a CPU NumPy reference that tests compare against.
  - **The design must be tuned carefully and inspected visually.** Success and failure must be distinguishable at a glance, as with MNIST strokes. There is a trade-off: a contrast nonlinearity makes shapes crisp but brings back aliasing. Contact sheets are required.
  - The tile banks remain available as an option, sampled with antialiasing.
- **Odometry and query encoding** live in one shared module,
  `mapping/se2c/encoding.py`, with `encode_odometry(dxdy, dtheta)` and
  `encode_query(dxdy, dtheta)`:
  - **Odometry:** the relative pose in the previous body frame, (dx, dy, dθ) with dθ in [-π, π).
  - **Query:** the body-frame offset from the anchor pose plus the heading change.
  - **Features:** translation as scaled Fourier features; the angle as [cos, sin] plus Fourier features at integer frequencies.
- **Queries: the hidden mine.**
  - **Contract.** Per-step causal coverage. Every target sample point must lie in the union of footprints, which are rotated squares, observed up to the query's step t, checked against first-seen times on a fine raster.
  - **Allowed variant.** `query_coverage="global"`: queries may lie anywhere covered by the whole sequence. If it is used, document it as a deviation in CONTRACT_SE2C.md.
  - **In both variants,** queries are placed right after observation t, take RoPE step t, and express their pose relative to the pose at t.
- **Evaluation:**
  - the floor
  - `mse_centered`
  - error binned by prefix length and by |turn|
  - an oracle-ceiling tool: true poses, reconstruct the field, render the query

  Solved thresholds are proposed lightly; the user says they will know it when they see it.
- **Geometry in fp32.**

## Workstreams (launched 2026-10-16, background agents)

1. **Environment and ports.** Phases, each committed before any real run:
   1. Geometry, texture, coverage, generator and encoding, with contact sheets.
   2. The GPU data path in train.py and render.py, plus CONTRACT_SE2C.md.
   3. The `hybrid4c` and `vanilla4c` ports, with a mechanism trace of how the turn angle reaches the delta rule's key and β, and throughput at 3x3 patches.
   4. A smoke run, then one local proving run of hybrid4c xs at 20-30k steps. Look at its grid.
2. **Blind probe (L3),** in `experiments/lift_blind_probe/`. Twists in, relative pose out, no pixels, using the shared encoding. Models: hybrid4-style, β capped at 1, attention only, optionally complex-diagonal. It measures:
   - pose error against length and granularity
   - the hero figure: held-out-angle eigenangle calibration, where slope 1 means a chart and a staircase means a lookup
   - β by angle
   - breakaway step against n
3. **Pre-freeze ablations (§06),** on discrete SE(2), with new model files under `mapping/models/se2/`. In order:
   1. A muP-consistent init for the short conv, as a new variant.
   2. hybrid4.5b, a fixed width-64 Gated DeltaNet coprocessor plus a growing attention+SwiGLU trunk, with the minimal-pair control (the same model minus the block).
   3. A placement sweep: first, middle, last.
   4. One against two coprocessors.
   5. A conv-width race, {1, 4, 10, 11} per head, only after the init fix.

   Results go into HYPOTHESIS.md amendments and `docs/lift/prefreeze_ablations.md`. Local runs for development; the cluster for anything over about 30 minutes.

## After the workstreams report

- If placement-insensitive, freeze L1-L5, the 4.5b ladder and the minimal pair into HYPOTHESIS-v2 with thresholds, then queue the granularity curriculum.
- If placement-sensitive, hybrid5 (a PaTH-hybrid) joins the frozen slate, and P6 becomes central.
- Then the scaled se2c runs on the cluster, following the scaling-study pattern.

## Horizon: SE(3) with a camera (framing, 2026-10-16)

**The objective is egocentric novel-view synthesis, end to end.**
- Render the view at a query pose from the observation and odometry/action history. Z², discrete SE(2) and continuous SE(2) are its rungs. The camera rung keeps the objective and inserts a projection.
- **The models are deliberately NOT streaming.** The memory is recomputed from the full history every forward pass, which is acceptable as long as the compute is parallel. hybrid4 scans the whole history in both directions. Incremental state updates are not a goal.
- **The user's distinction:**
  - VGGT-type models predict geometry (cameras, depth, point maps) feed-forward: a partial existence proof that general transformers learn camera geometry, at scale.
  - They do not, and presently cannot, do novel-view synthesis end to end.
- End-to-end NVS transformers exist (SRT, LVSM; details to check against the papers). They are conditioned on calibrated per-image poses.
- **LVSM's encoder-decoder** compresses posed views into latent scene tokens. The user notes it looks close to a scene memory a policy or world model could read end to end. A research agent is reviewing this: docs/lvsm_memory_review.md.
- **Our niche:**
  - egocentric, with relative odometry or actions in place of calibrated poses;
  - end-to-end NVS as the memory objective;
  - a geometric bias (exact state rotations from odometry) for efficiency.

**Proposed rungs.** Each has exact analytic rendering and lifts one thing at a time:
1. SE(2) motion with a pinhole camera over a textured ground plane: a homography; camera, no occlusion.
2. SE(3) motion over the same plane: roll and pitch enter the homography; group lift, no occlusion. Every 3D rotation is also a product of two reflections, so the delta-rule mechanism carries over. So does the topological seam (pi_1(SO(3)) = Z2).
3. 3D scenes with occlusion: true NVS.

**Baselines for the camera rung:**
- LVSM-style decoding, conditioned on the egocentric history: the vanilla analogue. The right conceptual family is a latent belief in activations, readable by free pose queries through the network (LVSM, SRT, CUT3R's state).
- VGGT-style models (explicit pointmap/geometry outputs) are a POOR fit for the program, however good their geometry. Their map lives outside the activations, and using it for decisions needs a renderer or geometry seam with restrictive inductive biases. At most a frozen visual front end. (User correction, 2026-10-16; see the Spatial POMDP Program Notes.)

**Prerequisites:**
- the continuous-SE(2) mechanism probe (calibration plus seam);
- episodes-to-threshold against vanilla4c;
- the patch-size decision;
- the Hydra branch merged.

## Open threads (2026-10-16, preliminary)

### Visual odometry

- **Discrete SE(2): works and resolves with scale.** Runs used no odometry (dropout 1.0), 3 px steps, high-passed Imagenette, 200k steps:
  - hybrid4 m: 0.0002;
  - hybrid4 s: 0.0034 (floor 0.0127);
  - odometry-given controls: m 0.0000, s 0.0004.
  
  High-passing removes the mean-brightness shortcut, which alone takes raw images from 0.072 to 0.046. xs failed at long range (capacity).
- **Continuous SE(2) (se2c): untested.** The generator has no odometry-dropout option yet, which is a small addition. The geometry favours it: steps of 4.5-9 px against 15 px views keep at least 40% overlap, and the stroke texture is registrable. Natural first run: hybrid4c s or m without odometry.
- **The user has design ideas for visual odometry,** to be elaborated before any run. Don't design it ahead of them.

### The penultimate lift: a 2.5D "ocean"

- **The user's vision:**
  - the ego glides on an SE(2) subspace at fixed altitude and pitch above a procedural height field h(x, y) (R²→R) carrying a procedural texture;
  - a pinhole camera gives perspective, parallax and real self-occlusion;
  - a step toward robot-POMDP realism (occlusion-unknowns, the near-deterministic doctrine), before full SE(3) plus 3D scenes.
- **Status:** a renderer prototype is in progress, with no task or trainer integration yet. Branch exp/ocean in ~/code/mapping-hydra, mapping/ocean/. It covers:
  - the band-limited height field with a slope bound;
  - the stroke albedo, with optional shading;
  - exact GPU ray-marching against a float64 reference;
  - anti-aliasing;
  - a visibility (depth-test) coverage check;
  - contact sheets and VIDEOS (the user needs motion to judge parallax).
- **Open design questions (the user's to decide):**
  - **Odometry:** SE(2) twist tokens as now, or something else once visual odometry is in play.
  - **Query poses:** the same altitude and pitch as the ego (pure SE(2) queries), or varied altitude and pitch (an SE(3)-lite readout).
  - **Coverage:** visibility-based causal coverage, meaning a target pixel counts only if its surface point was seen unoccluded earlier. How to treat partly covered targets: mask the loss on unseen pixels, or reject the query.
  - **Occlusion-unknowns:** represent "never seen" explicitly in the target or loss (known-unknown flags), or simply exclude it.
  - **Camera:** field of view, pitch, altitude, resolution and patch size. Token cost grows with resolution.
  - **Horizon and sky:** whether views see the horizon.
  - **Texture and relief:** the parameters, picked from the sheets and videos.
- **The patch-size and sample-efficiency decisions** carry over, since token counts only grow from here.
