# Merging refactor/hydra into master, and sealing the argparse path: review

Date: 2026-10-16. **Executed the same day; see section 9.**

Sections 0 to 8 are the review as written before the merge. It was done
on the scratch branch `merge/trial-hydra-into-master`, since removed.
Commit hashes such as `507ab51` refer to that branch. The same changes
are on master as listed in section 9.

## 0. Recommendation

Merge. The divergence is small: master has one code feature the branch
lacks (se2c odometry dropout) and one new tool (compare_grid). There is
one textual conflict. The trial merge, with the dropout ported into the
config objects, passes 458/458 tests, including bit-identical
equivalence against a pre-merge master checkout. The cases are z2,
se2d with WSD, se2c, se2c with dropout 0.5, and se2c visual odometry
(dropout 1.0), plus the GPU case within noise.

Seal the argparse path in two stages:
1. **At the merge.** `train.py --flags` stays a frozen translator. Its
   flag set is snapshotted by a test, so a new setting can never get a
   flag. It also prints a deprecation line.
2. **When the live Torch argparse work ends.** Delete the CLI. Keep the
   flag-to-override conversion only as a printer (`run.py --from-flags`),
   which never trains.

Do not delete argparse in one step. `train.build_parser()` is currently
the schema of the `args` record that every checkpoint and `final.json`
carries. It must first move to a declared record schema
(`mapping/record.py`), and the equivalence tests guard that move.

## 1. Divergence inventory

Merge base: `06cf83f` (STATUS: xs bridge not solved). Master HEAD:
`2c709c6`. refactor/hydra HEAD: `21aef11`.

**Master since the merge base (38 commits):**

| class | commits / files | action |
|---|---|---|
| docs only | STATUS (19 commits), lift plan (10), HYPOTHESIS H1c/H1d (3), causality literature review, LVSM memory review, multiview cycle-scan note, README report links | auto-merged, no action |
| code the branch must absorb | `038f19c` se2c `--odometry-dropout` (mapping/se2c/generator.py `Config.odometry_dropout`, `_drop_odometry`; task_se2c `config_from_args`) | **ported** (section 3) |
| code the branch must absorb | `5132a88`, `d663dd5`, `b737b3e` tools/compare_grid.py | auto-merged; works unchanged (se2c only, so the rename does not touch it; it uses `T.data(args)`, which keeps its signature); one open issue (section 3, C4) |
| job files | jobs/lift/h1c_grid.sbatch, h1c_tasks.txt, h1d_grid.sbatch, h1d_tasks.txt | auto-merged. They already call `run.py` (written for exp/causal on Torch), and they name `hybrid4_causal`/`vanilla4_causal`, which exist only on exp/causal (section 5) |
| conflicts with the rename or the config objects | only `038f19c`: it changed `config_from_args`, which Hydra rewrote. No master commit touches `se2`/`task_se2`/`models/se2`, train.py, render.py or summarize.py | resolved |

The brief named "se2c split metrics" and "render changes" as newer master
work. They are not after the merge base. `aad11b5` (split metrics by
query kind) and `6950723` (shuffled-odometry split) predate `06cf83f`, so
both sides already have them. No render.py commit exists on master after
the merge base.

Uncommitted work in `~/code/mapping` (master's working tree):
- `experiments/lift_blind_probe/probe.py` and `run_all.sh` (adds `flock`);
- untracked `docs/lift/prefreeze_ablations.md` and `docs/models/hybrid45b.md`.

None of these use the renamed names. They must be committed or stashed
before the real merge (section 7, step 1).

**refactor/hydra since the merge base (11 commits).** The commits run
from `ad61563` (rename se2 -> se2d) to `21aef11` (probe follow-ups):
- a016016: ladders to conf/;
- f98c20e: Hydra tree, run.py, train.py as translator;
- f389bcb: ladders as rungs;
- bd5f906: trainer consumes objects;
- 0e38feb: run safety, `--exp`/`--from`;
- eba37c0: flags/run.py agreement;
- 17afb20: tools read old se2 checkpoints;
- 2d55241: job generators write run.py, deps pinned;
- 640b4e9: docs/config.md.

The diff is 131 files, +4451/-601.

**exp branches.** Each is strictly ahead of refactor/hydra, none behind.
exp/odq currently equals refactor/hydra. It is left alone (another agent
is building it).

| branch | commits | contents | bring to main? |
|---|---|---|---|
| exp/causal | 3 (50341e4, 5f311b6, 23973e0) | models `hybrid4_causal`, `vanilla4_causal`, `hybrid4_mix` + 7 registered variants (conf/model/se2d/*), preset `se2d_causal`, tests (test_causal, test_causal_mix), tools/transition_probe.py one-direction support | **Yes, all of it.** H1c and H1d are registered hypotheses in master's HYPOTHESIS.md, and master's job files already run these models on Torch. Purely additive except transition_probe.py (backwards compatible: `scan_dirs` defaults to `both`). The causal models are byte-identical between 50341e4 (the Torch `LIFT_COMMIT`) and the tip, so Torch results stay reproducible from main. |
| exp/time-recurrence | 3 | models `hybrid4_time`, `hybrid4_time_full` + se2c ports `hybrid4c_time`, `hybrid4c_time_full`, preset `se2d_time_recurrence`, tools/time_scan_probe.py, tests, docs/time_recurrence.md (result: both variants FAIL at xs) | **Yes**: models, tests, probe and results doc. A negative result is a result, and the models are referenced by the doc. Additive, no shared file touched. Alternatively, merge the doc + probe only and leave the models on the branch. That is the user's call (section 8). |
| exp/ocean | 5 | mapping/ocean/ (height field, ray-march renderer, visibility), tools/ocean_sheets.py, tools/ocean_videos.py, tests/test_ocean.py, **97 MB of PNG/GIF/MP4** under docs/ocean/ | **Not yet.** The renderer's design questions are marked OPEN (user) in docs/lift/plan.md. The media would almost double the repository (`.git` is 135 MB). When the ocean rung is accepted: merge the code and tests, keep the media out of git (the sheets can be regenerated by tools/ocean_sheets.py; publish videos elsewhere). |
| exp/odq | 0 | in progress | untouched; rebase onto the merged main when it lands |

Trial (no commit, then aborted): an octopus merge of exp/causal,
exp/time-recurrence and exp/ocean onto the trial merge applies with no
conflict. The suite passes: **616 passed** (458 + 158 exp tests), with
MAPPING_REF at master HEAD.

## 2. Trial merge: result and test numbers

```
git -C ~/code/mapping fetch ~/code/mapping-hydra \
  refs/heads/refactor/hydra:refs/remotes/hydra/refactor/hydra 'refs/heads/exp/*:refs/remotes/hydra/exp/*'
git -C ~/code/mapping worktree add -b merge/trial-hydra-into-master ~/code/mapping-merge-trial master
git merge --no-ff hydra/refactor/hydra      # 1 conflict: mapping/task_se2c.py
```

Reference checkouts (detached worktrees):
- **merge base `06cf83f`.** The existing
  `.../scratchpad/hydra_ref`, the pre-refactor reference.
- **master HEAD `2c709c6`.** `.../scratchpad/master_ref`, the newest
  pre-merge master: se2c dropout, compare_grid.

| state | MAPPING_REF | result |
|---|---|---|
| raw merge, conflict resolved, nothing ported | merge base | **453 passed, 1 failed**: `test_cpu_runs_are_bit_identical[se2c]`. log.jsonl differs, because se2c runs on master since 038f19c record `generator.odometry_dropout: 0.0` and the merge-base reference does not. Expected: a master change, not a refactor change. |
| raw merge | master HEAD | equivalence 4/4 passed (z2, se2c, se2d_wsd CPU bit-identical; GPU hybrid4 within noise) |
| trial merge with the port (commit `507ab51`) | master HEAD | **458 passed, 0 failed** (3 min 24 s; GPU used only by the 20-step hybrid4 xs equivalence case and the existing generator tests) |
| trial merge with the port | merge base | equivalence: 3 passed (z2, se2d_wsd, GPU), 3 skipped (se2c cases: reference predates the knob, by design) |
| + exp/causal + exp/time-recurrence + exp/ocean (uncommitted, aborted) | master HEAD | **616 passed** |

New tests on the scratch branch (454 + 4 = 458):
- `tests/test_se2c_generator.py::test_odometry_dropout_zeroes_whole_steps_and_nothing_else`.
  Checks that p=0 is the unchanged stream, p=1 zeroes all odometry, and
  p=0.5 zeroes whole steps, with observations, targets, poses and query
  poses identical.
- `tests/test_config.py` FLAG_CASES gains the Torch VO run's command
  (`--task se2c --model hybrid4c --size m --steps 120000 --query-times uniform --odometry-dropout 1.0 --run vo_120k_qt`).
- The se2c tracked run sets `task.generator.odometry_dropout=0.5`. The
  trainer must read the new field, and the consumed-field test picks it
  up automatically.
- `tests/test_equivalence_runs.py` gains `se2c_dropout` (0.5) and
  `se2c_vo` (1.0) cases, bit-identical against master HEAD: log.jsonl,
  final.json, checkpoints and grid.png.

Manual smoke (CPU, 4 steps):
- run.py with `task.generator.odometry_dropout=1.0` writes
  `final.json` with `odometry_dropout 1.0`, `shuffle_informative false`
  and `generator.odometry_dropout 1.0`.
- `run.py --print` shows the exact train.py equivalent.
- compare_grid.py runs on two such checkpoints.

## 3. Conflicts and resolutions

**C1, textual: `mapping/task_se2c.py::config_from_args`.**
- Master added `odometry_dropout=a.get('odometry_dropout', 0.0)`.
- Hydra rewrote the function to pass through every `Config` field that
  the record carries.
- Resolution: take Hydra's pass-through, plus
  `same.setdefault('odometry_dropout', 0.0)` for records older than the
  field. Every record since fbfad8a carries the key anyway, because it
  was a bank flag with default 0.0.

**C2, semantic: se2c dropout was not a config field.** On refactor/hydra:
- `train.py --task se2c --odometry-dropout 1.0` is refused ("does not
  apply to this run"). Verified on the branch tip.
- `task.generator.odometry_dropout` does not exist for se2c, so it is
  refused at instantiate.

So the running Torch VO baseline (`mapping_lift/results/se2c/hybrid4c/m/vo_120k_qt`,
argparse train.py at 038f19c) could not be reproduced from the branch.
The failure is loud, not silent. Port (commit `507ab51`):
- `Se2cGenerator.odometry_dropout: float` in mapping/config.py, and
  `odometry_dropout: 0.0` in conf/generator/se2c.yaml.
  `Se2cGenerator.config()` already passes every field of the same name
  to `generator.Config`, so it is consumed with no other code.
- `legacy_args()` copies it to `args.odometry_dropout`, so the record is
  as master writes it, and the `--branch` check on `odometry_dropout`
  works for se2c.
- train.py `FLAG_KEYS['odometry_dropout']` changes from `'bank'` to
  `None` (both generators).
- `shuffle_informative = g.odometry_dropout < 1.0` on every task. On
  master it was `args.odometry_dropout < 1.0`; the branch had
  `gpu_data or ...`. This is a record and print field only, but it is
  in final.json, so the equivalence test covers it.
- Not done, flagged: train.py's `--odometry-dropout` help text and
  CONTRACT_SE2C.md still describe only the bank meaning ("reserved
  unknown symbol"). On se2c it is a zero vector. Docs fix at the real
  merge.
- Not done, flagged: the bank generator validates `0 <= p <= 1`; the
  se2c generator (on master too) does not. Adding the check is a
  behaviour change on master's side (section 8).

**C3, equivalence reference.** From 038f19c on, se2c records carry
`generator.odometry_dropout`. So against the merge-base reference, every
se2c equivalence case fails on that one key; this is master's change,
not the refactor's.
- Resolution: the se2c cases require a reference at or after 038f19c
  and skip otherwise, with that reason. The z2/se2d cases run against
  either reference.
- The evidence for the merge is the master-HEAD run (all cases). The
  evidence for the refactor alone remains the merge-base run.

**C4, compare_grid with odometry dropout (open, not changed).**
- `TASK_KEYS` does not include `odometry_dropout`. Checkpoints with
  different dropout are therefore compared silently, and the queries are
  drawn with the first checkpoint's settings: verified, it ran on a p=0
  and a p=1 checkpoint without complaint.
- A VO model then sees real odometry it never trained on, or an odometry
  model sees zeros.
- Either refuse (add the key to `TASK_KEYS`), or deliberately allow
  "same episodes, each model its own dropout" (draw once, apply each
  checkpoint's dropout to its own copy). That is a design decision
  (section 8).

**C5, rename vs compare_grid.** No change needed:
- compare_grid refuses non-se2c tasks;
- `load_checkpoint` returns canonical task names;
- the task key it compares is `se2c` on both sides.

**C6, job files on master** (jobs/lift/h1c_*, h1d_*):
- They need `hybrid4_causal`/`vanilla4_causal`, so they only compose
  once exp/causal is merged.
- They read `jobs/h1c_tasks.txt`, but master stores the file as
  `jobs/lift/h1c_tasks.txt`. They are records of what Torch ran from
  `/scratch/ax2348/mapping_hydra` (where the file sits at `jobs/`), not
  launchers for this tree.
- Keep them as records; note the path at the merge, or move them to
  `jobs/torch/` with a header comment.

## 4. Who depends on the argparse path

| dependent | what it runs | effect of sealing |
|---|---|---|
| Torch `/scratch/ax2348/mapping` (old tree, no git) | 32 sbatch files, `train.py --task se2 ...`; results/{z2 models, se2} | none: its own frozen copy of the code. Its results are read by main through the `se2` alias (`summarize.py --task se2`, `load_checkpoint`) |
| Torch `/scratch/ax2348/mapping_lift` (LIFT_COMMIT 038f19c) | 37 job/launch files with `train.py --task se2c ...`, the tmux worker (commands typed by hand), the live VO run vo_120k_qt | none while the tree is not re-synced. **Re-syncing main into it after the CLI is deleted would break every typed command.** Its 3 run.py job files are copies of mapping_hydra's |
| Torch `/scratch/ax2348/mapping_hydra` (LIFT_COMMIT 50341e4, exp/causal) | run.py only (h1c, h1d, hydra_causal_400k); jobs 20070043 running, 20072852 and 20076519 pending | none; already Hydra |
| Torch `/scratch/ax2348/mapping_odq` | does not exist yet | none |
| scaling.py, tools/branch_jobs.py | generate run.py commands (since 2d55241) | none |
| run.py `--print` | prints the train.py equivalent via `train.flags`/`overrides_from_flags` | must change (print the run.py line only) |
| mapping/config.py `legacy_args()` | builds the record from `train.parse` + `train.build_parser()._actions` (defaults and types) | **load-bearing**: the record schema; must move, not disappear |
| tests/test_config.py (FLAG_CASES, refusal, translation tests), tests/test_equivalence_runs.py (`flags` leg; the `old` leg runs the reference's own train.py) | translator | translator tests go with the CLI; the old-vs-run.py leg is the evidence and stays |
| summarize.py, render.py, mapping/results.py, tools/*.py (probes, audit, compare_grid) | read the `args` dict from checkpoints / final.json; none imports the parser | none, as long as the record keeps its keys |
| old checkpoints / final.json (`task: 'se2'`, no `arch`) | `tasks.ALIASES`, `load_checkpoint` fallback to the module's `build` | none; covered by test_old_se2_checkpoints_load_as_se2d, test_summarize_reads_every_task_and_the_old_name |
| results paths | new se2d runs go to results/se2d; old ones stay in results/se2 (alias root) | none. Do not move results/se2: final.json paths and docs link to it |
| README "Training and evaluating", CONTRACT*.md, docs/config.md, STATUS/plan prose | show `train.py --flags` commands | rewrite to run.py at stage 2; dated STATUS/plan entries stay as history |
| experiments/lift_blind_probe/train.py | a separate, self-contained probe trainer (not the harness; no results/ run directory) | out of scope; the seal's test must allowlist `experiments/` explicitly |
| tools/bench.py, tools/se2c_bench.py | throughput benchmarks with their own argparse (backward passes, no runs) | out of scope; allowlist |

## 5. Sealing the legacy path

Options considered:
- **Remove argparse train.py entirely, now.** This breaks `legacy_args`
  (the record schema), `run.py --print`, and the translator tests.
  Nothing on Torch breaks immediately, but the next sync of
  `mapping_lift` does. It is too much in one step, and it gains nothing
  until the live argparse work ends.
- **Deprecated translator that refuses new flags.** Essentially what
  refactor/hydra already is: exact or refused. It still needs a freeze,
  so that nobody adds a flag for a new setting (the se2c dropout port
  above re-used an existing flag; it did not add one).
- **Keep only for old cluster job files.** Old job files run their own
  tree's code. Main never executes them, so main needs only a
  *converter* (flags -> run.py overrides) to re-run one, not a trainer.

**Proposed seal (two stages):**

**Stage 1, with the merge. Freeze.**
1. `tests/test_seal.py::test_train_py_flags_are_frozen` asserts that the
   set of `train.build_parser()` option strings equals a committed
   snapshot (`tests/legacy_flags.json`, the 40 flags as of the
   merge). A new flag fails CI. A new setting is a config field: the
   record picks it up through `legacy_args()` with no flag (already
   true).
2. `main_from_flags` prints one stderr line, as the first output:
   `train.py --flags is frozen; equivalent: python run.py <overrides>`.
   It still trains, through the same config tree, so it stays exact.
3. `test_only_run_py_trains`: an AST scan of `*.py` outside
   `experiments/` and `tests/`. It asserts that `train.main(` is called
   only from run.py and from train.py's own `main_from_flags`, and that
   no module other than train.py builds an `ArgumentParser` with both
   `--model` and `--steps`. Allowlist: tools/bench.py,
   tools/se2c_bench.py, experiments/**.

**Stage 2, when nothing live uses argparse.** That is, when the Torch VO
run and any typed `train.py` worker commands are finished, and
mapping_lift is retired or its launchers are rewritten.
1. Move the record schema out of train.py. `mapping/record.py` holds
   `LEGACY_FIELDS = {dest: (default, type)}`, generated once from the
   frozen parser and committed. `legacy_args()` builds the Namespace from
   it. The equivalence tests (old train.py vs run.py, bit-identical
   final.json and checkpoints `args`) prove the record did not change.
   This is the step that needs them.
2. Delete `build_parser`/`parse`/`flags`/`FLAG_KEYS`/`main_from_flags`
   from train.py. `python train.py ...` exits non-zero: "train.py is the
   trainer library; runs go through run.py (docs/config.md). To convert
   an old command: python run.py --from-flags '<flags>' --print".
3. Keep the flag-to-override table as `run.py --from-flags` (print-only,
   or compose then run, at the user's choice). It reads old job files
   and old `final.json` `args` and prints the run.py overrides. It is
   tested on the frozen FLAG_CASES plus every distinct command in the
   Torch job files (copied into the test as strings).
4. Enforcement replaces stage 1's tests:
   - `test_no_argparse_trainer`: train.py has no `argparse` import, and
     `python train.py --model x --size y` exits non-zero with the
     pointer.
   - The AST scan from stage 1 stays.
   - `test_legacy_record_is_frozen`: `LEGACY_FIELDS` equals its
     snapshot, plus keys added only through config fields.
5. Equivalence tests: the `flags` leg goes. The `old` leg keeps running
   the reference checkout's own train.py against run.py, so the
   bitwise-equivalence evidence survives. Pin the references as tags:
   `ref/pre-hydra` = 06cf83f and `ref/pre-merge` = 2c709c6. Then
   MAPPING_REF can always be recreated with
   `git worktree add --detach <dir> ref/pre-merge`.
6. Rewrite README "Training and evaluating", CONTRACT*.md flag mentions
   and CONTRACT_SE2C "train.py --task se2c takes" in run.py terms.
   Dated STATUS/plan entries stay as written.

An import guard (raising on `import argparse` inside train.py) is weaker
than the tests above and makes train.py's import fragile. Prefer the
tests.

## 6. Risks to behaviour and reproducibility

- **se2c dropout before the port.** Loud, not silent: refused by both
  entry points. After the port it is bit-identical to master (se2c_vo
  and se2c_dropout equivalence cases).
- **Reference choice hides a change.** Against the merge-base reference,
  se2c fails on `generator.odometry_dropout`. Anyone re-running the
  evidence must use `ref/pre-merge` (or later) for se2c. The skip makes
  this explicit rather than a mystery failure.
- **Old se2c records through `config_from_args`.** These are used by
  render.py, compare_grid and the tools. A record with a nonzero
  `odometry_dropout` from *before* 038f19c would now have dropout
  applied when redrawn, although training ignored it. I checked every
  se2c `log.jsonl`/`final.json`, locally (results/se2c) and on Torch
  (mapping_lift/results/se2c). The only nonzero value is vo_120k_qt,
  trained at 038f19c with the knob live. So there is no affected run
  today. The two Torch `se2/.../seed0_200k_qt_vo_hp` runs are bank-task
  VO, where dropout was always live.
- **Re-syncing Torch trees.** mapping_lift runs argparse train.py at
  038f19c, with a hand-typed tmux worker. Syncing main there after
  stage 2 breaks those commands. Syncing after stage 1 changes their
  results directory for se2d only, which mapping_lift does not train.
  mapping_hydra runs exp/causal at 50341e4; main after the merge plus
  exp/causal reproduces it (the causal model files are unchanged since
  50341e4).
- **Pinned dependencies.** hydra-core 1.3.5 and omegaconf 2.3.1 are new
  requirements for main (requirements.txt; Torch job files check them).
  The local env has them. Any other machine that runs main needs
  `pip install -r requirements.txt`.
- **`results/se2` vs `results/se2d`.** Same task, two roots. summarize.py
  needs `--task se2` for the old and `--task se2d` for the new ones.
  Mixing them in one fit requires both. Do not move directories.
- **Repository size.** Merging exp/ocean as is adds 97 MB of media.
- **Cleanup of this trial.** Fetching the exp branches into
  `~/code/mapping` added their objects (including ocean's media) under
  `refs/remotes/hydra/*`. Remove them with
  `git -C ~/code/mapping update-ref -d` for each ref, then
  `git worktree remove` for the trial and master_ref worktrees, and
  `git gc` if wanted.

## 7. Merge plan (step by step), with rollback

1. **Settle master's tree.** Commit or stash the uncommitted
   lift_blind_probe changes and the two untracked docs. Tag the
   references: `git tag ref/pre-hydra 06cf83f`,
   `git tag ref/pre-merge master`.
2. **Port onto the branch.** Cherry-pick the port into refactor/hydra
   in ~/code/mapping-hydra (or merge master into refactor/hydra and
   apply the same resolution). This keeps the exp branches' base
   correct. The diff is commit `507ab51` of this branch minus the merge
   itself:
   - mapping/config.py, conf/generator/se2c.yaml, train.py,
     mapping/task_se2c.py;
   - the 3 test files.
3. **Fix the docs.** The `--odometry-dropout` help text,
   CONTRACT_SE2C.md (the se2c dropout as a generator setting, zero
   vectors), and the jobs/lift note (section 3, C6).
4. **Run the suite twice.** `MAPPING_REF=<ref/pre-merge worktree>`: all
   pass, incl. se2c_dropout and se2c_vo. `MAPPING_REF=<ref/pre-hydra
   worktree>`: all pass, se2c cases skipped. The GPU equivalence case is
   short (20 steps, xs), so run it when the local GPU is free.
5. **Merge into master.** In ~/code/mapping:
   `git merge --no-ff refactor/hydra`. The only conflict is
   task_se2c.py, resolved as here. Re-run step 4 on master.
6. **Merge the exp branches.** Merge exp/causal, then
   exp/time-recurrence (no conflicts in the trial; 616 tests pass).
   Leave exp/ocean and exp/odq on their branches, and rebase or merge
   main into them.
7. **Stage 1 seal.** Commit the frozen-flags snapshot test, the
   deprecation line and the AST scan, as one commit on master.
8. **Stage 2 seal, later.** After the Torch argparse runs end, as a
   separate PR: record schema to mapping/record.py; delete the CLI; add
   `run.py --from-flags`; update the tests and docs.

**Rollback.**
- Before step 5 nothing on master has changed.
- After step 5, `git reset --hard ref/pre-merge` on master is safe while
  nothing has been pushed or synced. Afterwards use
  `git revert -m 1 <merge>`, which keeps history; re-merging later then
  needs `git revert <the revert>` first.
- Steps 6 to 8 are separate commits, each revertable alone.
- Torch is unaffected by any of this, because each tree is a copy
  stamped by `LIFT_COMMIT`.
- Results directories are never touched by the merge.

## 8. Open decisions for the user

1. **Seal timing.** Stage 1 now and stage 2 after the VO run and the
   mapping_lift worker finish (recommended)? Or delete the CLI at the
   merge and accept that mapping_lift must not be re-synced from main?
2. **`run.py --from-flags`.** Print only, or allowed to run? Printing
   only is the stricter seal.
3. **compare_grid with mixed odometry dropout.** Refuse it? Or support
   "same episodes, each model its own dropout" (C4)?
4. **se2c `odometry_dropout` range check** (`0 <= p <= 1`, as the bank
   generator has). Master's se2c generator accepts any value; p > 1
   behaves as 1, and p < 0 as 0.
5. **exp/time-recurrence.** Merge the models too, or only the results doc
   and probe (the models failed)?
6. **exp/ocean.** Keep it on the branch until the OPEN renderer
   questions are settled? And where should the media live (outside git)?
7. **Master's jobs/lift/h1c and h1d files.** Keep them as Torch records
   with a header, or move them to `jobs/torch/`?
8. **Reference tags.** May `ref/pre-hydra` and `ref/pre-merge` be created
   on master's repository (tags only, no branch change)?

## 9. Executed (2026-10-16)

Decisions taken:
1. Seal stage 1 now; stage 2 later.
2. compare_grid warns.
3. Range check on odometry dropout.
4. exp/time-recurrence: all of it.
5. exp/ocean: code and tests only.
6. Jobs layout as below.
7. Tags.
8. Cleanup of the trial.

Master commits, oldest first:

| commit | what |
|---|---|
| 6ba7dbb | settle: lift blind probe code and figures (runs/ gitignored, untracked) |
| 1c526aa | settle: docs/models/hybrid45b.md. The untracked docs/lift/prefreeze_ablations.md (empty template) and transformer-mapping-study.md (stray pointer) were deleted |
| tags | `ref/pre-hydra` = 06cf83f (branch point); `ref/pre-merge` = 1c526aa (master after settling, before the merge). Both annotated |
| dd8f2c7 | merge refactor/hydra (conflict in task_se2c.py resolved as C1) |
| e820edf | se2c odometry dropout as a consumed generator field (C2, C3; the tests of section 2) |
| 23aa318 | merge exp/causal (all) |
| 959e94e | merge exp/time-recurrence (all) |
| d111c53 | exp/ocean 3569764 squashed in, code and tests only. Its 97 MB of media are in `results/ocean_media/` (gitignored); docs/ocean/README.md says where and how to regenerate them. No media blob is in master's history (checked with `git rev-list --objects`). The tools' `--out` default is still `docs/ocean/...`; the README says to pass `--out results/ocean_media/...` |
| 2606b66 | (coordinator) STATUS: odq VO array submitted, docs only |
| c701547 | decision 3: odometry_dropout outside [0, 1] is refused with a message (both config generators via `Generator.__post_init__`, the se2c `Config`, train.py flags); test |
| 5be7b4e | decision 2: compare_grid compares any differing task setting (the se2c generator Config each record rebuilds, plus TASK_KEYS) only with a boxed stderr warning and an inverted banner in the image's title area; a different task is still refused; tests/test_compare_grid.py |
| 51ac0d4 | decision 6: jobs/ tracked next to conf/experiment/ (layout in jobs/README.md); lift launchers committed; new job logs under results/jobs/logs/ (scaling.py, tools/branch_jobs.py, remote_setup.sh) |
| 7fe0fc7 | seal stage 1: tests/test_seal.py + tests/legacy_flags.json (40 flags with defaults), the deprecation line in `train.main_from_flags`, and the scan that only run.py calls `train.main` |
| 49d6c97 | scaling.py --slurm makes jobs/ itself (a regression from 51ac0d4 the suite caught) |

**Jobs layout (decision 6).**
- `conf/experiment/` holds what a run is.
- `jobs/<campaign>/` (tracked) holds how it was submitted.
- `results/jobs/logs/` (untracked) holds the SLURM output of new jobs.

The launchers are kept exactly as submitted, Torch paths included.
- Older logs stay in the gitignored `jobs/logs/`.
- The 30 pre-Hydra root launchers (z2/se2, argparse) moved to the
  gitignored `jobs/legacy/`.
- Open: track jobs/legacy/ as provenance of results/se2 and the z2 runs.
- There are no odq launchers on exp/odq or in ~/code/mapping-odq to move.

**exp/odq: not merged.** It carries its own, independent port of the
same field. In 26beba8:
- `Se2cGenerator.odometry_dropout`, the same line in
  conf/generator/se2c.yaml;
- `FLAG_KEYS[...] = (None, ...)`;
- `shuffle_informative = g.odometry_dropout < 1.0`;
- a generator.py identical to master's apart from the new range check;
- its own equivalence/config test edits.

The semantics agree. Rebase exp/odq onto the new master and keep
master's version of each overlapping hunk:
- config.py and se2c.yaml: the field and the range check;
- train.py: the FLAG_KEYS line and shuffle_informative;
- generator.py;
- tests/test_equivalence_runs.py, which needs `ref/pre-merge`.

odq's own additions stay: `Objective`, the odq models and presets, and
the shuffle_informative override for odometry-query models. That
branch's owner must do this, after checking that the Torch runs launched
from it (array 20084102) do not depend on its tip staying put.

**Verification on merged master (49d6c97).**
- `MAPPING_REF=<worktree of ref/pre-merge>`: **623 passed, 0 failed,
  0 skipped**. All 6 equivalence cases passed: CPU bit-identical for z2,
  se2d_wsd, se2c, se2c_dropout and se2c_vo; the GPU hybrid4 xs case within
  noise.
- `MAPPING_REF=<worktree of ref/pre-hydra>`: test_equivalence_runs gives
  3 passed (z2, se2d_wsd, GPU) and 3 skipped (the se2c cases, which need a
  reference at or after 038f19c).
- Recreate a reference worktree with
  `git worktree add --detach <dir> ref/pre-merge` and link data/ into it.

**Stage 2 of the seal (not done).** Section 5 describes it. Do it when
the Torch argparse work (mapping_lift worker, vo_120k_qt) has ended. It
is reversible by reverting its commit.
