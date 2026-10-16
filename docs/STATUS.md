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
| Lift environment and ports (DONE 2026-10-16, report in docs/lift/build_log.md) | se2c task, texture, coverage, encoding, hybrid4c / vanilla4c, proving run | mapping/se2c/, mapping/task_se2c.py, mapping/models/se2c/, CONTRACT_SE2C.md, docs/lift/build_log.md, minimal hooks in train.py, render.py, tasks.py |
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
- **Old sweep 19991387:** the original `hybrid` model (pre-hybrid2) on DISCRETE SE(2), day 1's 40k-step lr ladder with last-step queries. The user confirmed 2026-10-16 that it stays.

After runs land: `scripts/sync.sh pull`, then `python summarize.py --task se2`,
tools/audit_se2.py on near-zero checkpoints, and the analysis plan in
docs/scaling_study_design.md.

## Open decisions for the user

- Decision-arm `[decide]` items (docs/decision_arm_design.md).
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

## Autonomous session: SE(2) proper (started 2026-10-16, ~6 hours, user-approved)

**Plan:**

1. Steer the three lift agents as they report:
   - review the texture sheets, coverage, the proving-run grid and the probe calibration myself;
   - launch the follow-ups each implies;
   - keep docs/lift/ and the journal current.
2. Once hybrid4c trains, run the granularity curriculum (n = 4, 8, 16, 32, continuous) and look at breakaway against n (L1). Then vanilla4c as the matched comparison, then the probes on what solves.
3. Read the scaling study as it lands: threshold crossings, cooldown branches into fits, audits of near-zero checkpoints. Record results as dated amendments.

**Boundaries:**

- Do NOT freeze HYPOTHESIS-v2; draft it for the user's sign-off.
- Do not cancel jobs I did not submit (the discrete-SE(2) `hybrid` sweep 19991387 stays).
- **Lift and ablation work is LOCAL only, with training jobs SEQUENTIAL across all agents.** Enforced with a shared lock:
  `flock /tmp/claude-1000/-home-adrianna/b0ebca23-53ce-4dc1-a5c5-072191a4dabe/scratchpad/gpu_train.lock <train command>`.
  Data work, tests and smoke runs are exempt. The scaling study stays on the cluster.
- Never edit the user's lift notes or the frozen hypothesis text; amendments only.
- Keep all the working rules above.

**Reporting:**

- one row per milestone in the journal's day-2 log;
- a wrap-up for the user covering what worked, what didn't, and the decisions waiting.

**Session log**

- **2026-10-16, lift build done** (commits d8b0da7, f4d73d8, e68c17f, 97a8229, ed4eb63; 209 tests pass).
  - I reviewed texture.png, episode.png and the proving grid by eye. The stroke texture reads at a glance, and success and failure are obvious per row.
  - Proving run, hybrid4c xs at 25k: 0.0655 (shuffled 0.0715, floor 0.131). Not yet solved; the discrete xs needed ~90k.
  - **Open for the user (flagged, not filled):**
    1. With the default omega in U[-pi, pi) the walk is diffusive: it stays in a ~40 px blob, and queries sit a median 10.8 px from the anchor (episode.png shows it). A persistent walk needs either a smaller omega_max plus heading-aligned query proposals, which changes the query distribution, or `--motion decoupled`. Your call; the default is unchanged.
    2. MSE on sharp strokes does not grade near misses: a 2 px / 10 deg error already scores at the floor. A blur-tolerant diagnostic metric would be my addition to the contract; not added.
    3. The near band-pass texture spectrum deviates from the suggested ~1/f (the lift agent's choice, documented).
  - **Registered before results: a seam.** The lift agent predicts that a continuous-angle hybrid4c cannot rotate exactly at every theta. I checked the argument.
    - Suppose a step's transition is a product of reflections, I - 2kk^T, whose keys vary continuously with theta.
    - That product is the image of a continuous path in Pin(d).
    - The full-turn loop in SO(2), a subgroup of SO(d), is the nontrivial element of pi1(SO(d)) = Z2, so it does not lift to a closed loop in Pin(d).
    - So some theta must break the pure-reflection form: beta != 2 or decay < 1, which contracts.
    - Prediction: there is an angle band, probably near +-pi, with larger error and a contracting transition there. The n-lattice settings escape this, since no continuity is needed.
  - **Queued (one launcher, under the lock), hybrid4c xs at 100k:**
    1. `bridge_n4dec_100k_qt`: `--motion decoupled --turn-n 4 --v-min 9 --v-max 9`, the discrete task's odometry statistics with the new renderer, texture and encoding. It tests the port separately from the group.
    2. `seed0_100k_qt`: the default continuous task.
- **2026-10-16, user decision: the decoupled walk is the se2c default** (commit aafa32f).
  - Moves: directions change by U[-3pi/4, 3pi/4) per step, the same at every n. v ~ U[4.5, 9].
  - Turns: i.i.d. on [-pi, pi) or the n-lattice.
  - Odometry: the canonical relative pose, the turn wrapped to [-pi, pi).
  - Effect: median end displacement 40 px and query offset 15 px, against 31 and 10 px for the arc walk. Floor 0.1307.
  - The queued arc default run was cancelled before it started. Queued instead, under the lock:
    1. `bridge_n4dec_100k_qt` (`--directions lattice --turn-n 4 --v-min 9 --v-max 9`)
    2. `dec_100k_qt` (the new default)
- **2026-10-16, the bridge moved to Torch** (user's call): job 20036564. It runs from a separate tree, /scratch/ax2348/mapping_lift, so the scaling study's code stays frozen while its jobs are queued. Job file: jobs/lift/se2c_bridge_hybrid4c_xs.sbatch (jobs/ is gitignored). Locally, `dec_100k_qt` (the new default) is next under the lock, and the ablation queue waits for it.
- **2026-10-16, PAUSE, user's call: focus only on the bridge -> new-default hill climb.**
  - Both agents (pre-freeze ablations, blind probe) are stopped, along with their launchers. Paused, not cancelled:
    - convrace xs, killed at 78k of 100k (checkpoints every 5k; I first misreported this as ~35k);
    - hybrid45b_nocopro, _mid and _two at s;
    - the blind probe's cdiag and its n-sweep. Its gdn_b1 was reported solving: long-lag position nmse 0.017 at 68k.
  - The Torch bridge job 20036564 was cancelled. The bridge runs locally now, alone on the GPU.
  - The next run, the new default, is launched by hand after the bridge is read.
- **2026-10-16, blind probe SHELVED.**
  - Why: without pixels there is no stored map to transport, so the task loses the pressure that makes an exact state rotation pay off. The user agreed it is "a fairly limited idea in the first place".
  - Recorded results, 100k steps each, lag 17-32 position nmse:
    - GDN, beta in (0, 2): 0.0068
    - GDN, beta capped at 1: 0.0085
    - attention: 0.091
  - GDN beta stayed ~0.47 (l2r) and ~0.09 (r2l), flat in the turn, with no rotating plane: no reflection route.
  - Held-out turn band = control band, so no held-out failure.
  - L3's calibration and seam questions move to the transition probe on hybrid4c in the real se2c task.
  - Code and runs stay in experiments/lift_blind_probe/ (uncommitted run outputs are not tracked).
- **2026-10-16, capacity test of the bridge on Torch** (user's call): hybrid4c s 20038399 and m 20038403. Same bridge settings, latest code (split metrics), checkpoints every 5k. Reason: the xs bridge's far-query bin crawls (0.84 of floor at 50k) while fallbacks at 36 px are rendered correctly. The user suspects visual capacity at xs.
- **2026-10-16, bridge hybrid4c xs, 100k steps (local): NOT solved.**
  - Eval mse overall 0.0706: floor 0.1317, shuffled 0.1027.
  - By kind, error/floor:
    | kind | error/floor | note |
    | --- | --- | --- |
    | fallback | 0.12 | shuffled 0.51 |
    | near | 0.16 | |
    | mid | 0.32 | |
    | far (>= 15 px, 58% of queries) | **0.79** | |
  - Grid (docs/lift/sheets/bridge_hybrid4c_xs_100k_grid.png):
    - fallback views 36-38 px away are crisp, and wrong with shuffled odometry;
    - near-anchor queries are right;
    - far novel views go from grey to a coarse light/dark layout, never strokes.
  - Reading: pose composition works; assembling a novel view from stored content is the bottleneck at xs. The user suspects visual capacity: the continuous texture plus 3x tokens per view is too much for xs.
  - Next: the s bridge on the Torch worker (20038588), and m (20038403).
  - To make room, every pending scaling task was cancelled and one running lr check (20028844_1) was stopped. All are recorded in docs/scaling_study_design.md.
