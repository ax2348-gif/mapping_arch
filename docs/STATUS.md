# Project status and handoff (updated 2026-10-16)

The single place to resume from after a context reset. The research journal
holds the narrative:
https://claude.ai/artifact/24DMrX4JZkcwD4DGthpLYG (Claude Docs doc 088bfc9f-b37f-4622-b6da-c2ec7c39fe39,
prose node 579ba254-c943).

## Where things stand

- **Discrete SE(2).** hybrid4 solves it at xs (0.0180, 2 seeds). Its trained
  right-to-left delta-rule head rotates a 2-D subspace by exactly the turn, and
  the rotation composes over 16 steps. Ablations removing β > 1, the conv, or
  non-commutativity all fail at xs, 100k steps.
- **Overnight surprise.** vanilla4 (all attention) also solves SE(2) at scale:
  l meets the threshold at 50k steps, xl at 30k, against hybrid4 l at 10k. P1 is
  effectively falsified. H survives only in its "much more efficient" form.
  See the HYPOTHESIS.md amendments.
- **Visual odometry** works on high-passed Imagenette and resolves with scale
  (hybrid4 m 0.0002).
- **The three arms:** decision and memory (on hold; docs/decision_arm_design.md,
  LANav as the scaled test); multi-agent mapping (proposed; regimes R1-R3, port
  to marlenv); the Lie lift (ACTIVE; docs/lift/plan.md).

## In flight

**Background agents,** each committing only its own files:

| Agent | Scope | Writes to |
| --- | --- | --- |
| Lift environment and ports | se2c task, texture, coverage, encoding, hybrid4c / vanilla4c, proving run | mapping/se2c/, mapping/task_se2c.py, mapping/models/se2c/, CONTRACT_SE2C.md, docs/lift/build_log.md, minimal hooks in train.py, render.py, tasks.py |
| Blind probe | L3: eigenangle calibration | experiments/lift_blind_probe/ |
| Pre-freeze ablations | conv muP init, hybrid4.5b and minimal pair, placement, 1 vs 2 coprocessors, conv-width race | mapping/models/se2/ (new files), HYPOTHESIS.md amendments, docs/lift/prefreeze_ablations.md, cluster jobs |

**Torch cluster** (`ssh torch`; check with
`squeue -u ax2348`; a status script is at /scratch/ax2348/mapping/peek.py):

- **Scaling study** (docs/scaling_study_design.md, predictions S1-S4):
  - A: long runs 20014708 (vanilla4 l, xl), 20014709 (hybrid4 l), 20028595 (vanilla4 xs, s, m), 20028596 (hybrid4 xs, s, m), 20028840 (hybrid4 xl)
  - B: 64-step ladders 20028842 (hybrid4), 20028843 (vanilla4)
  - C: cooldown branches 20028849-20028855 (afterany on their parents)
  - D: learning-rate check 20028844, 20028846
  - E: second seed 20028847, 20028848
- **Visual odometry, high-pass texture, s and m, 200k:** 20012146 (done), 20012147 (odometry given; m was running).
- **Old SE(2) hybrid sweep 19991387.** It predates hybrid2 and still occupies a slot. The user was asked repeatedly whether to cancel it and has not answered; leave it.

After runs land: `scripts/sync.sh pull`, then `python summarize.py --task se2`,
tools/audit_se2.py on near-zero checkpoints, and the analysis plan in
docs/scaling_study_design.md.

## Open decisions for the user

- Decision-arm `[decide]` items (docs/decision_arm_design.md).
- Whether to cancel the old SE(2) sweep.
- Multi-agent arm: a design note, after reviewing the multi-robot SLAM literature.

## Working rules learned (also in the agent memory)

- **One local training run per launcher.** Stop the launcher, not only the
  trainer, and verify with `ps -eo args | grep "[p]ython train.py"`.
- **Before any model-facing change, write a mechanism trace.** A minimal diff is
  not a port. Use assertions, not eyeballing.
- **Never edit HYPOTHESIS.md above "Amendments".** Add dated amendments.
- **Flag any rule I add to a contract as mine.**
- **Cluster:**
  - H200 jobs take at most 16 CPUs per GPU.
  - Triton needs Python headers (scripts/pyhdr.sh, already applied).
  - Texture banks need a manual rsync.
  - Conda's default channels require accepting the Anaconda Terms of Service; use conda-forge.
- **Commits** end with the attribution lines given in the session. Commit by explicit path.

## Published pages (all private until shared)

| Page | Link |
| --- | --- |
| research journal | https://claude.ai/artifact/24DMrX4JZkcwD4DGthpLYG |
| hybrid4 results report | https://claude.ai/artifact/NjMCowkCEio9mcYQ1tF8Ti |
| primer | https://claude.ai/artifact/9jFX4Cr2761JFTCBytKPQ3 |
| Z² scaling | https://claude.ai/artifact/8mjpJKB9LUMMKipyHtkRNK |
| theory review | https://claude.ai/artifact/DJ4WFQ5V7ArJrDV31EQ8bN |
| applications review | https://claude.ai/artifact/WnVhdCjzUtq1VRo6mySS8A |
| literature synthesis | https://claude.ai/artifact/MSr6KNebLT7Rttep2nnU1H |
| user's lift notes (reference) | https://claude.ai/artifact/LUSCC1MToip91haQkC28zc |
