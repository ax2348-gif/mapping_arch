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
| se2d scaling report (preliminary, 2026-10-16) | https://claude.ai/artifact/KGG23hvdWGZ7pDpiPQ7qbc |
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
- **2026-10-16, capacity was the bottleneck: the s bridge breaks away.** Torch worker 20038588, at 37.5k: far 0.0192 (0.15 of floor; shuffled 0.164), all 0.0142. xs ended at 0.79 of floor on far. The local m bridge is at 10k: far 0.51 of floor.
  - **Plan, made official by the user:** the moment the s bridge ends, launch hybrid4c s on the default continuous task (`dec_100k_qt`, jobs/lift/launch_dec_s.sh on the cluster) into the same worker. Then analyse the bridge (final split, grouped grid) and intervene if needed.
- **2026-10-16, Hydra refactor started** in ~/code/mapping-hydra, branch refactor/hydra (background agent). Design approved by the user:
  - no unconsumed config fields (defaults-list composition, instantiate);
  - se2 renamed se2d;
  - model configs per task, size ladders as groups;
  - experiment yamls (`+experiment`, `--exp path`);
  - a resolved config.yaml per run, rerun with `--from`;
  - legacy flat `args` kept as an output;
  - bitwise behaviour tests.
- **2026-10-16, s bridge stopped at ~47.5k (user's call: the default-task run has higher information gain).**
  - Last eval, 47.5k: all 0.0086, last-step 0.0178. By kind, true | shuffled:
    | kind | true | shuffled |
    | --- | --- | --- |
    | fallback | 0.0028 | 0.084 |
    | near | 0.0055 | 0.046 |
    | mid | 0.0086 | 0.118 |
    | far | 0.0111 | 0.17 |
  - Every grid row at 45k is correct by eye, far included (docs: results/se2c/hybrid4c/s/bridge_n4dec_100k_qt/grid_step45000_kinds.png, local).
  - **My error:** a `pkill -f` pattern inside worker 20038588 also matched the batch script, so the worker ended and its GPU was released.
  - New worker 20041129 has the default-task s run (`dec_100k_qt`) as its first task.
  - Resume the bridge later, same cosine schedule, from the 45k checkpoint (on the cluster):
    `train.py --task se2c --model hybrid4c --size s --steps 100000 --query-times uniform --eval-every 2500 --log-every 500 --save-every 5000 --motion decoupled --directions lattice --turn-n 4 --v-min 9 --v-max 9 --branch results/se2c/hybrid4c/s/bridge_n4dec_100k_qt/step45000.pt --run bridge_n4dec_100k_qt_from45k`
- **2026-10-16, data scaling on the continuous task (user's call):** hybrid4c m, default se2c, 400k WSD, checkpoints every 25k, as a regular Torch job: 20042194 (jobs/lift/se2c_dec_hybrid4c_m_400k_wsd.sbatch, code 8837c28). Reason: a real robot's SE(2) cannot be discretised, so data and size are the levers to study, not the n-curriculum. The live session keeps iterating on the worker.
  - Also running: worker 20041129, hybrid4c s on default se2c at 100k. At 32.5k: far 0.086 (0.66 of floor), shuffled 0.132.
- **2026-10-16, local m bridge stopped at ~22.5k** (user's call: the local GPU goes to the Hydra seams probe). Last eval at 22.5k: all 0.0061. By kind, true | shuffled:
  | kind | true | shuffled |
  | --- | --- | --- |
  | fallback | 0.0018 | 0.076 |
  | near | 0.0037 | 0.037 |
  | mid | 0.0054 | 0.103 |
  | far | 0.0082 | 0.157 |
  - m passes the far bin faster than s: s's far was 0.019 at 37.5k. Last checkpoint: results/se2c/hybrid4c/m/bridge_n4dec_100k_qt/step20000.pt (local).
- **2026-10-16, m bridge on Torch (20039511), cancelled at ~65k (user's call).**
  - It was a resubmission the user had rejected; the submit had run before the rejection arrived. Cancelled as soon as noticed.
  - Result at 65k: all 0.0008, last-step 0.0015. By kind, true | shuffled:
    | kind | true | shuffled |
    | --- | --- | --- |
    | fallback | 0.0003 | 0.081 |
    | near | 0.0006 | 0.057 |
    | mid | 0.0006 | 0.126 |
    | far | 0.0010 | 0.159 |
  - So m solves the bridge essentially exactly. The size trend on the bridge: xs far 0.79 of floor at 100k; s 0.0111 at 47.5k; m 0.0010 at 65k.
- **torch-gpu (~/torch-gpu):** found that `tmux capture-pane` crashes the tmux server on Torch's tmux next-3.4, on the login node too. That explains the earlier failed tests. The tool now reads session output from its pipe-pane log; retest pending.
- **2026-10-16, se2d scaling report (preliminary) published:** https://claude.ai/artifact/KGG23hvdWGZ7pDpiPQ7qbc
  - **S1 supported:** the T=32 compute multiplier at matched loss is 2.8-5.3x from 0.02 to 5e-4, and 7-10x from 0.07 to 0.04.
  - **S2 not supported:** the multiplier shrinks as loss falls.
  - **S3 supported at s only:** at T=64 hybrid4 s reaches 0.0011, while vanilla4 s ends at 0.037.
  - **S4 untestable:** the reference branch was cancelled.
  - Audits pass (hybrid4 l, vanilla4 xl, hybrid4 s T=64).
  - A draft HYPOTHESIS amendment is in the report, NOT applied: it needs user approval.
- **2026-10-16, hybrid4c s on default se2c, 100k (worker): final 0.0260** (shuffled 0.0906, floor 0.1309). By kind:
  | kind | error | note |
  | --- | --- | --- |
  | fallback | 0.0040 | |
  | near | 0.0056 | |
  | mid | 0.0140 | |
  | far | 0.0447 | 0.34 of floor, still falling: data-limited |
  - 75k grid: correct at arbitrary continuous angles except the farthest queries (~56 px).
- **Matched vanilla on se2c (user priority):**
  - vanilla4c s 100k cosine: 20043638
  - vanilla4c m 400k WSD: 20043639
  - hybrid4c m 400k WSD runs in worker 20041129 (`dec_400k_wsd_worker`). The duplicate queued hybrid job 20042194 was cancelled (user's yes).
  - The weight-decay / grokking-like study is shelved behind se2c.
- **2026-10-16, se2c matched results so far:** at 100k,
  | model | all | far | note |
  | --- | --- | --- | --- |
  | hybrid4c s | 0.0260 | 0.0447 | |
  | vanilla4c s | 0.0633 | 0.1082 | barely uses odometry: shuffled 0.0764 |
  - hybrid4c s reached vanilla4c s's 100k score by ~20k: ~5x fewer episodes. One seed.
  - hybrid4c m 400k (worker) at 65k, stable phase: 0.0282, far 0.0492.
- **2026-10-16, se2c ladder extension queued (user):** array 20046841, jobs/lift/se2c_dec_ladder.sbatch. All WSD, checkpoints every 25k:
  - hybrid4c and vanilla4c l at 200k;
  - hybrid4c and vanilla4c s at 400k (the m runs' data budget).
  - Plus vanilla4c m 400k (20043639), running.
- **Small-model policy (user):** Torch kills low-utilisation jobs (hybrid4 xs seed 1, 20028847_0, was killed at 360k; its 350k checkpoint survives). Roll small jobs together; spend Torch on bigger models.
- **Hydra branch:** all 13 seams fixed (8 commits). 454 tests pass, re-run independently. The cluster venv needs `pip install -r requirements.txt` before use.
- **2026-10-16, experiments queued by the user (discrete SE(2), xs, through Hydra on branch exp/time-recurrence off refactor/hydra):**
  1. **Symmetry, running (agent):**
     - `hybrid4_time`: the recurrence sees only odometry tokens (DeltaProduct, 2 Householders, β ∈ (0, 2)) and read-only query tokens, forwards and backwards; its outputs are added back to patches. The last layer is regular attention.
     - `hybrid4_time_full`: the same layer count, with the last layer a full GDN over all tokens.
     - Reference: hybrid4 xs, 0.0180, solved at ~65k.
  2. **Causal, NEXT, after 1 finishes:** a fully causal hybrid4 (forward-only GDN, causal attention, matched shape) plus a causal vanilla4 control.
     - Key question: the rotation was found in the RIGHT-TO-LEFT head (P5 on left-to-right failed), so can it form in the forward direction?
     - Probe the forward heads.
- **2026-10-16, symmetry pair (discrete, xs, Hydra branch exp/time-recurrence): both fail.**
  | model | eval at 100k | shuffled |
  | --- | --- | --- |
  | hybrid4_time | 0.0526 | 0.0916 |
  | hybrid4_time_full | 0.0487 | 0.1071 |
  | hybrid4 | 0.0180 | |
  - The time-only scan in hybrid4_time DOES learn the exact rotation from odometry alone: an eigenpair within 10° of the turn on 100% of quarter- and half-turn steps, β median 1.69.
  - Reading: what's missing is the associative content memory the rotation carries in hybrid4.
  - Caveat: at xs (2 layers) the _full variant has no attention at all. The clean A/B (last layer only) needs s, which is deferred.
  - Write-up: docs/time_recurrence.md on that branch.
- **Causal experiment started (user chose it before the s A/B pair):** branch exp/causal off refactor/hydra; hybrid4_causal and vanilla4_causal at xs; sequential, local.
- **2026-10-16, FINDING (provisional): fully causal hybrid4 fails where bidirectional succeeds** (se2d xs, matched shapes, branch exp/causal).
  - Causal at 90k: 0.0575, shuffled 0.0879. Bidirectional: 0.0186, shuffled 0.1403. The curves split at ~20k, and causal never breaks away.
  - **The information confound is ruled out on last-step queries,** where both models see identical history: causal last-step 0.072 vs bidirectional 0.026 at 100k. The gap is memory quality, not access to future observations.
  - The user's reading:
    - bidirectional context is cheap wherever the history is recomputed anyway (history -> 1 action, history-conditioned rendering), and may be why it was overlooked;
    - it breaks teacher-forced world models (Dreamer-style), so bridging causal vs bidirectional is a research question in itself;
    - it harmonises with recent NVS work.
  - **Caveats:** one size (xs) and one seed. The vanilla4_causal control and the forward-head probe are pending.
  - **Follow-ups proposed:**
    - causal hybrid4 at s (impossible, or just costlier?);
    - a causal backbone plus one final bidirectional delta-rule layer, or chunk-bidirectional;
    - where the rotation forms.
