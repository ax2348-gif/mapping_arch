# Hybrid7 full-weight SE2D → SE2C transfer

User requested training transfer from the completed register-only Hybrid7
AGAGA S100k model. Target is Hybrid7c S: identical1,155,105 learned parameters,
five registers, six GDN heads, A,G,A,G,A. Only the task adapter changes from
discrete symbols/quarter turns to continuous local poses. All weights load
strictly, with a fresh optimizer, schedule and step0; no partial transfer.

Preset: `+experiment=hybrid7_se2c_transfer`. Standard continuous SE2C,
15x15 procedural views, T32,8 uniform-time queries, continuous rotations,
decoupled persistent moves4.5–9px, causal coverage, batch64, IEEE scans.
100k new steps, LR1e-4, warmup100, WSD decay80–100k. Lower learning rate is a
fine-tuning choice, informed by earlier SE2C LR comparisons, not an optimum
established for Hybrid7. Eval every2500 over1024 held-out episodes includes
far-query and shuffled controls; checkpoints every10k.

Source: `results/se2d/hybrid7/s/hybrid7_causal_inert_saw_s_agaga_100k_ieee_seed0/final.pt`.
Target: `results/se2c/hybrid7c/s/from_se2d100k_agaga_s_100k_ieee_seed0`.
Launcher: `results/hybrid7_se2c_launch/launcher.log`, tmux `hybrid7-se2c`.
No odometry-writer variant is used or launched.

The warm-start harness allows only the specific Hybrid7 SE2D→Hybrid7c SE2C
adapter-class pair with identical constructor arguments. Other architecture
or task mismatches are still rejected. Source hash/run ID and adapter transfer
are recorded in provenance. CPU adapter equivalence and warm-start regression
tests passed (27 tests). GPU preflight checks exact loaded weights and evaluates
zero-shot SE2C before any optimizer updates; results in
`results/hybrid7_se2c_transfer_preflight/results.json`.

This experiment changes task texture, view size, motion distribution and turn
continuity together. It tests practical transfer to standard SE2C, not an
isolated explanation of any transfer difficulty.

Zero-shot evaluation (1024 held-out episodes): intact MSE0.154025,
shuffled0.153939, far0.156773, last-step0.151117. No measurable odometry
advantage before adaptation. All transferred tensors were bit-exact.

Launched run `20261019T030918Z-erid-2-a15e03`, snapshot `396d55046eff`.
The source final.pt omits a checkpoint step field (the startup log prints
`step None`); its final.json and source run verify100k completed steps.

## MNIST retention at SE2C step10k

Frozen before/after comparison via `mnist_retention.py`,128 matched held-out
episodes per task, seed0, batch8,16batches; strict full-weight adapter loading.
Saved `results/hybrid7_mnist_retention/step10000/results.json`. Evaluated in
small GPU batches alongside training; training was not stopped or modified.

| Task | Original weights MSE | SE2C10k MSE | SE2C10k shuffled |
| --- | ---: | ---: | ---: |
| Original SE2D MNIST9 SAW |0.013297|0.085160|0.108120|
| Continuous SE2C MNIST15 |0.086496|0.034308|0.062912|

Original-task last-step MSE worsens0.01969→0.09816. Continuous MNIST far
MSE improves0.09166→0.05226. The checkpoint retains useful MNIST-texture
processing on the new continuous15px task, despite training on procedural
textures, but strongly regresses on original discrete9px trajectories.
Thus this is not a universal inability to process MNIST; view size, motion
and trajectory-distribution changes remain confounded with forgetting.
No claim of preserved SE2D performance, and no joint-training change launched.

## Discrete9px SAW with expanded15px MNIST views, SE2C10k checkpoint

Clarified follow-up: preserve the original SE2D trajectory distribution and
quarter turns, expand only views. `se2c10k_discrete_view15_eval.py` uses strict
full-weight loading into the discrete input adapter and the same256 episodes,
seed1808/halo1809 as the pre-transfer view-size evaluation. Saved output:
`results/hybrid7_se2c10k_discrete_view15_controls/{results.json,renders.png}`.

| Observation → query | Center9 MSE | Shuffled center9 |
| --- | ---: | ---: |
|9→9|0.087379|0.111018|
|15→15|0.065648|0.093850|
|15→9|0.065603|0.093230|
|9→15|0.087500|0.112670|

Full15→15 MSE0.065888 versus shuffled0.092570. Before transfer, the same
condition scored0.085420 full /0.041849 center. Thus larger observations
partially recover performance relative to9px inputs in the adapted model;
query expansion alone does not. Full-view average improves from better outer
regions, but center retrieval worsens relative to the original checkpoint.
These findings support view-size adaptation as part of the regression, not a
complete explanation. Discrete15px error remains substantial and the saved
preselected examples are often blurry or incorrect. Training remains unchanged.

## User-requested automatic M follow-up (prepared; launch blocked)

Prepared `hybrid7_se2c_medium`: fresh M, width192, AGAGAGA,3,269,581 parameters,
six GDN heads, same continuous SE2C settings and100k/LR1e-4 budget. S weights
cannot be loaded into M without partial transfer, so M starts from scratch.
`queue_se2c_medium.py` waits for successful S completion, verifies final outputs,
and uses a lock/launch marker to prevent duplicates. Intended tmux session:
`hybrid7-se2c-medium-queue`. Two launch attempts and an automation creation
attempt were NOT executed because automatic approval review was at capacity.
No automatic start is confirmed. Do not claim this follow-up is scheduled until
queue execution or automation creation succeeds through normal approvals.

## Final S render/geometry diagnostic

S completed100k successfully: held-out MSE0.022864, far0.037263,
last0.047621, shuffled0.098289. `render_se2c_geometry.py` evaluates128
held-out episodes (1024 queries), saves arrays, enlarged target/prediction/
contour/error comparisons, and simple oracle fits in the final run's
`geometry_diagnostic/`. Rows explicitly select median and90th-percentile MSE
within each query kind; they are not random or best-case examples.

Visual reading: fallback/near contours mostly coincide; mid cases increasingly
soften and distort; far cases show substantial blur with some displaced or
incorrect boundaries. This supports substantial geometric success but not
perfect geometry with only cosmetic smoothing remaining.

For far queries, fixed-center9 MSE0.02943 becomes0.02023 after per-query best
translation of prediction in +/-2px at0.5px intervals (31% lower). Comparing
prediction to an optimally Gaussian-blurred target gives0.01297 (56% lower).
Median best shift0.5px; median blur sigma1px. These oracle fits are overlapping
explanations, not a decomposition of error or decoded pose estimates; blur can
also reflect uncertainty/mixing of several wrong spatial addresses.

The user's interrupted medium launch DID start PID872711 before interruption.
On the subsequent explicit request to inspect S first, medium training was
paused with SIGSTOP (verified process state T), preserving its in-memory state.
Do not resume it automatically. Queue status may still say running because the
queue is waiting for that stopped child. No heartbeat was successfully created.

## Final SE2C100k checkpoint: discrete SE2D variants

`se2cfinal_discrete_view15_eval.py` repeats the exact256-episode matched
MNIST SAW9px/quarter-turn protocol (seed1808/halo1809), with strict full-weight
discrete adapter loading. Results/renders:
`results/hybrid7_se2cfinal_discrete_view15_controls/{results.json,renders.png}`.

| Observation → query | Pre-transfer center9 | SE2C10k center9 | SE2C100k center9 | Final shuffled center9 |
| --- | ---: | ---: | ---: | ---: |
|9→9|0.015051|0.087379|0.089054|0.100392|
|15→15|0.041849|0.065648|0.047392|0.091593|
|15→9|0.042017|0.065603|0.049078|0.091605|
|9→15|0.015644|0.087500|0.088491|0.100394|

Full15→15 MSE improves0.085420 pre-transfer →0.065888 at10k →0.047898
at100k (final shuffled0.093644). Aligned final-time retrieval remains weaker:
15→15 center0.071193, full0.068566, shuffled full0.099160.

Conclusion: meaningful partial recovery on15px observations as continuous
training progresses, with a strong odometry advantage and improved borders.
Original9px observations do not recover. Changing query size alone does not
rescue them. Supports adaptation/specialization to larger history views, but
motion/texture/coverage differences and incomplete learning remain confounded.
Saved examples still include wrong/blurred retrieval; not a solved SE2D15 task.
Medium training remains paused; no new training or architecture changes.
