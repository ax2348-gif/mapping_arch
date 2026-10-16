# Experiment provenance: design (Phase 2, for approval)

Status: PROPOSAL, 2026-10-16. Nothing here is implemented. §0b (build, adopt or hybrid) decides first; the sections after it assume its recommendation, BUILD. It follows the brief the user agreed,
the audit in `experiments/audit/README.md` (Phase 1), and the user's later decisions (A)
run-directory template, (B) the `xp reproduce` layout, and (C) a code-structure audit.
Decisions I added myself, not in the brief, are marked **[mine]**. Open questions are collected
at the end.

## 0. What the audit found, and the requirements that follow

| audit finding | requirement |
| --- | --- |
| 0 of 159 runs record their code. 86% have only an inferred commit. No run records whether its tree was dirty. | Every run pins an immutable commit of the exact bytes that ran, before training starts. |
| Torch trees are rsync exports with no `.git`. Each push overwrote the code under queued jobs. `LIFT_COMMIT` is a hand-kept, overwritten file. | Torch runs from real git; a queued job's code cannot change under it. |
| 44 local runs have no launch record. Pre-Hydra sbatch files are untracked. | The command, config, host and job ids go into a per-run record that is versioned. |
| One results path holds two different runs, and the next `sync.sh pull` destroys one. 9 stale mirrors. | Runs are identified by id, not path. Tools look a run up in the ledger rather than reconstructing its path. |
| Report figures were built from scratchpad copies. | Reports cite run ids, whose records live in git (§10, open question). |
| exp/odq code exists only in one clone. | Snapshot refs are pushed to a second machine. |

## 0b. Build, adopt, or hybrid (decide this first)

**The facts.** These were checked on 2026-10-16 against primary sources:
- release history from the PyPI JSON API (`pypi.org/pypi/<pkg>/json`);
- repository state from the GitHub REST API;
- behaviour from each tool's own docs.

Items I could not confirm are marked UNVERIFIED.

**Maintenance status of the candidates:**

| tool | latest release | maintenance | sources |
| --- | --- | --- | --- |
| DVC | 3.67.1, 2026-03-31 | Acquired by lakeFS (Treeverse), announced 2025-11-18. The repo moved iterative/dvc -> treeverse/dvc, not archived. The last default-branch commits are dependabot bumps (2026-08), so maintenance looks low-tempo. Docs moved to doc.dvc.org. | pypi.org/project/dvc; lakefs.io/blog/lakefs-acquires-dvc/; dvc.org/blog/dvc-joins-lakefs-your-questions-answered/; github.com/treeverse/dvc |
| Guild AI | 0.9.0, 2023-02-25 | Effectively unmaintained: last commit 2025-04 (a domain move), and my.guildai.org no longer resolves. | pypi.org/project/guildai; github.com/guildai/guildai |
| MLflow | 3.17.1, 2026-10-13 | Very active. The filesystem tracking backend (`./mlruns`) is deprecated: since 3.7 the default is `sqlite:///mlflow.db`, and `./mlruns` stays opt-in. | github.com/mlflow/mlflow/issues/18534; mlflow.org/docs/latest/self-hosting/ |
| W&B | 0.31.0, 2026-10-16 | Active. Owned by CoreWeave since 2025-05-05; docs now at docs.coreweave.com. | coreweave.com/blog/coreweave-completes-acquisition-of-weights-biases |
| Sacred | 0.8.7, 2024-11-26 | Slow but alive (last commit 2025-10). | sacred.readthedocs.io/en/stable/observers.html |
| exptrack | 2.1.0, 2026-10-02 | New, 0 stars. | github.com/mikylab/exptrack |
| mthd -> logis | 0.2.1 / 0.3.1, 2025-02 | Dormant since 2025-02. | github.com/flywhl/logis |
| expctl | 0.3.0, 2026-07-23 | New, 0 stars. | github.com/CactysFedya/expctl |
| expmonkey | 0.0.35, 2024-05-03 | Dormant. A worktree navigator, not a run tracker. | github.com/liuyibo/expmonkey |

**What each one does that matters here:**

- **DVC experiments.**
  - **Storage.** Experiments are commits under hidden custom refs (`.git/refs/exps`), based on
    `HEAD`, never checked out and not pushed by default. They are shared with `dvc exp push/pull`.
    (doc.dvc.org/user-guide/experiment-management)
  - **Untracked files are left out.** "Only files tracked by either Git or DVC are saved to the
    experiment." The exception is `dvc exp save --include-untracked`.
    (doc.dvc.org/command-reference/exp/run, .../exp/save)
  - **Snapshots without a pipeline.** `dvc exp save` snapshots the workspace with no pipeline. The
    docs do not say it needs a dvc.yaml.
  - **Pipeline required to run.** `dvc exp run` runs "a DVC experiment based on a DVC pipeline",
    from dvc.yaml stages. A run without any stage is UNVERIFIED, but in practice a stage
    (`cmd: python run.py`) is needed.
  - **Hydra.** It is opt-in (`hydra.enabled`). DVC composes `conf/` itself and dumps the result to
    `params.yaml`, and "existing Hydra code must be refactored to read params.yaml". The compose
    API and launchers are not mentioned. The same page warns that `dvc exp run` "is not a reliable
    way to reproduce past experiments".
    (doc.dvc.org/user-guide/experiment-management/hydra-composition)
  - **Queue.** The queue is dvc-task plus Celery workers on one machine, running in temporary copies
    under `.dvc/tmp/exps`. There is no Slurm support in any doc checked.
- **Guild AI.**
  - It makes a source copy per run (`sourcecode` config) in a run store under `~/.guild` (the run
    store location is UNVERIFIED; the docs site did not load).
  - It has a Hydra example, but flags must be declared in `guild.yml`.
  - Its remotes are ssh, ec2, s3 and azure. There is no Slurm remote.
- **MLflow.**
  - It tags `mlflow.source.git.commit`. That the classic tags record no dirty flag or diff is
    UNVERIFIED. Only the experimental GenAI versioning records a diff.
  - Viewing needs `mlflow ui` or `mlflow server`, a local process.
- **W&B (offline).**
  - `WANDB_MODE=offline`, then `wandb sync`. It records the HEAD sha and a `diff.patch` of
    uncommitted changes (code saving is off by default at the org level).
  - Viewing needs W&B cloud or a self-hosted server. A serverless local viewer is UNVERIFIED.
- **exptrack.**
  - Stdlib only. It keeps one SQLite file, with the commit, the full diff against HEAD, and a
    content-addressed script copy.
  - It has a Slurm "run-finish" mode, but no hidden refs and no Hydra.
- **mthd/logis.** It auto-commits on the current branch for every run: the commit spam we
  rejected.
- **expctl.** A SQLite plus JSON record keeper. The README does not say how it pins code.
- **expmonkey.** One directory per branch, with an fzf navigator. It is not a tracker.
  - The idea is relevant to our multi-agent juggling: today `~/code/mapping-hydra`, `-odq`, `-ocean`
    and `-seams` are hand-made worktrees.
  - But `git worktree` plus a naming rule (`xp workspace new <name>` -> `~/code/mapping-<name>`,
    branch `exp/<name>`) is a few lines of porcelain. It does not need a dependency.

### Requirement by requirement

Key: ✔ meets it as is; ◐ partly, or with work against the grain; ✘ no.

| requirement | DVC exp | Guild | MLflow (file/sqlite) | W&B offline | exptrack | build (this design) | hybrid: DVC `exp save` + ours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| immutable commit per run, no branch spam (hidden refs) | ✔ `refs/exps` (untracked only with `-I`) | ◐ a file copy, not a commit | ✘ a sha tag only | ◐ sha + patch | ◐ sha + diff in SQLite | ✔ `refs/runs/<id>`, untracked included | ✔ `refs/exps` via `exp save -I` |
| ledger syncs through git without conflicts | ◐ exps travel as refs, but the metadata lives in exp commits, not a reviewable ledger | ✘ home-dir store | ✘ a DB or dir outside git | ✘ cloud | ✘ a SQLite file does not merge | ✔ one JSON per run, per-host directories | ◐ our ledger, DVC's refs |
| Hydra compose API and our run-safety rules (non-empty dir refusal, `--from` drift check, strict objects) | ✘ DVC composes itself into params.yaml, bypassing run.py | ◐ flags in guild.yml | ✔ (agnostic; we log) | ✔ (agnostic) | ✘ argparse capture | ✔ run.py unchanged | ✔ if only `exp save` is used |
| Slurm / Torch submission, ~/torch-gpu | ✘ local Celery queue only | ✘ no Slurm remote | ✘ none (we would build it) | ✘ none | ◐ Slurm env capture only | ✔ `xp submit`, `xp gpu` | ◐ ours to build anyway |
| agent-friendly CLI (`--json`, `--explain`) | ◐ `dvc exp show --json`; no explain | ◐ | ◐ Python API | ◐ | ◐ JSON export | ✔ designed in | ◐ |
| reproduce at `<run>/reproductions/<ts>_<id>/` | ✘ runs in `.dvc/tmp/exps` or the workspace; the docs advise `dvc repro` instead | ✘ own run store | ✘ | ✘ | ✘ | ✔ `run.out` override | ◐ ours to build |
| run-directory template `${run.root}/...` | ✘ (its outputs are stage `outs`) | ✘ own run dirs | ✔ agnostic | ✔ agnostic | ✔ agnostic | ✔ | ✔ |
| light deps, no server | ◐ no server, but a heavy dependency tree (dvc + dvc-task + celery + kombu ...; exact tree UNVERIFIED for 3.67) | ✔ but dead | ✘ the UI needs a server process; sqlite default | ✘ cloud or server to view | ✔ stdlib | ✔ stdlib + git | ◐ dvc dependency for one feature |
| maintenance risk | medium-high: ownership change, slow cadence | high: unmaintained | low, but it is moving away from files | low, but commercial and cloud-tied | high: new, 0 stars | ours, on git plumbing that has been stable for over 15 years | DVC's risk for ~20 lines of value |

### Recommendation: BUILD the thin layer, borrowing DVC's design and not its code

1. **The only feature DVC offers that we want is hidden refs.**
   - That is about 15 lines of git plumbing (§2: `add -A` into a temporary index, `write-tree`,
     `commit-tree`, `update-ref`).
   - Adopting `dvc exp save` would add a heavy dependency, a `.dvc/` directory and `dvc init` to
     both machines, for the one step that is easiest to write.
   - It would still leave untracked files out unless flagged, and the project is in a slow
     post-acquisition phase.
2. **DVC's Hydra path conflicts with ours.** It composes the config itself and hands the program a
   `params.yaml`. That bypasses `run.py`, the sealed single entry, and its safety rules: the
   non-empty-directory refusal, the `--from` drift check, the strict config objects. It also
   requires a dvc.yaml stage per run.
3. **Nobody covers Slurm, the held-GPU tool, or the agreed reproduce layout.** Those, plus the
   ledger and `--json`/`--explain`, are most of the work under any option, so adopting a tool saves
   little.
4. **The ledger needs plain JSON in git, one file per run.** A tool whose store is SQLite (exptrack,
   expctl, MLflow's new default), a home directory (Guild) or a cloud (W&B) fails this requirement
   by construction.
5. **What is borrowed** is DVC's `refs/exps` idea (as `refs/runs`), Sacred's `run.json` field set,
   W&B's offline-then-sync workflow, and MLflow's one-directory-per-run file layout (as the export
   target, open question).

**Honest maintenance estimate for BUILD:**
- **Code.** `mapping/provenance.py` about 300 lines, `mapping/xp/` about 900-1,200 lines,
  `tests/test_provenance.py` plus the xp tests about 700 lines.
- **Time.** About 2 working days of agent time, plus a migration day for Torch (§6).
- **Ongoing cost.** It is driven by our own changes: new subcommands, and Slurm template drift.
  - The git plumbing it rests on (`commit-tree`, `update-ref`, `GIT_INDEX_FILE`) has not changed in
    over 15 years.
  - The risky parts are ssh and Slurm edge cases, which the fake-ssh and fake-sbatch tests (§9)
    cover.
  - If it rots, the data is still plain git refs and JSON, readable with `git` and `jq`.

**Consequences of each option for migration and integration:**
- **Build** (recommended): §2-§12 stand as written.
- **Adopt DVC.**
  - `dvc init` in the repo, and `dvc.yaml` with a `train` stage (`cmd: python run.py ${...}`).
  - `hydra.enabled` would make DVC compose `conf/`, so run.py must read `params.yaml`: a change to
    the sealed entry, and the bitwise equivalence tests would need re-proving. Alternatively, keep
    DVC's Hydra off and pass overrides through the stage cmd, losing `-S`.
  - Untracked files need `-I` discipline.
  - Slurm, the ledger, reproduce and the agent layer are still ours to build.
  - Reproducing via `dvc repro` conflicts with the agreed `reproductions/` layout.
  - Result: more integration work than building, plus a dependency.
- **Hybrid** (DVC `exp save -I` as the snapshot primitive, everything else ours).
  - Feasible. Refs land in `refs/exps/...` instead of `refs/runs/...`, and `xp sync` pushes that
    namespace.
  - Cost: the dvc dependency on Torch and locally, and a `.dvc/` directory.
  - Gain: about 15 lines not written.
  - I would choose it only if the user values "an established tool's ref format" above dependency
    weight. Then §2 changes to `dvc exp save -I -n <run_id>`, and the ledger records the exp ref.
- **MLflow or W&B as the store.** Both would replace the ledger and drop "syncs through git". They
  remain possible later as one-way export targets from the ledger (`xp export`), with no effect on
  this design.
- **Guild, mthd/logis, expmonkey, exptrack, expctl.** Not viable: dormant, the wrong model (commit
  spam), not a tracker, or too new to depend on.

## 1. Run identity

`run_id = <UTC yyyymmddTHHMMSSZ>-<host>-<6 hex>`, for example `20261016T201503Z-torch-3fa9c1`.

- **The parts.** The time sorts, the host names the ledger directory, and the hex comes from
  `os.urandom`. Two array tasks starting in the same second cannot collide.
- **Why the id carries no model or name [mine].** Names are mutable and already collide (the m
  bridge). The model, size and name are ledger fields, so `xp ls` shows them.
- **Valid as a ref name.** It is valid as `refs/runs/<run_id>` (`git check-ref-format`): no `:`, no
  `..`, no spaces.
- **`host` is a logical name, not a node:**
  - `torch` for any Torch node;
  - `erid-2` locally (the `XP_HOST` env var, else `socket.gethostname()` mapped through a small
    table in `conf/hosts.yaml`) **[mine]**.

  One directory per logical host keeps the single-writer property: only Torch jobs write
  `ledger/torch/`.

## 2. Snapshot refs

When a run starts, `run.py` (not `xp`, so the guarantee holds however a run is launched) does the
following, before any data, model or optimiser step:

```sh
top=$(git rev-parse --show-toplevel)                 # refuse if this fails: "not a git checkout"
idx=$(git rev-parse --git-dir)/xp-index-$run_id      # a private index, same filesystem
cp "$(git rev-parse --git-dir)/index" "$idx"         # seeds the stat cache (fast); contents get replaced below
GIT_INDEX_FILE=$idx git add -A -- . ':(exclude)experiments/ledger'
                                                     # tracked + untracked, .gitignore respected
tree=$(GIT_INDEX_FILE=$idx git write-tree)
base=$(git rev-parse HEAD)
if [ "$tree" = "$(git rev-parse HEAD^{tree})" ]; then
    snap=$base                                       # clean: the run IS its base commit
else
    snap=$(GIT_AUTHOR_NAME=run.py GIT_AUTHOR_EMAIL=run@localhost \
           GIT_COMMITTER_NAME=run.py GIT_COMMITTER_EMAIL=run@localhost \
           git commit-tree "$tree" -p "$base" -m "run $run_id (dirty snapshot of $base)")
fi
git update-ref -m "run $run_id" "refs/runs/$run_id" "$snap" ""   # "" = must not exist yet
rm -f "$idx"
```

- **Never touches the branch, HEAD or the real index.** Only objects and one new ref are written.
  The test in §8 checks `HEAD`, the index checksum and `git status --porcelain` before and after.
- **Clean trees reuse HEAD [mine].** The ref points at an existing commit, so a clean run's
  snapshot is the commit everyone already knows. That is also what makes `--reproduce` and Torch
  submits (§4, §6) yield the same sha as the original.
- **The fixed identity** means compute nodes with no `user.name` cannot fail. The commit message
  carries the base.
- **The dirty check** is `tree != HEAD^{tree}` over the same pathspec. It ignores
  `experiments/ledger/`, as the brief says. The ledger is also left out of the snapshot's
  *content* [mine]: a run's code never includes other runs' records.
- **Ignored files are not captured.** `results/`, `data/*.npz` and `*.pt` stay out. Data is pinned
  by hash in the ledger instead (§3, `data`).
- **Size guard [mine].** If any untracked, non-ignored file in the snapshot is over 20 MB, the run
  refuses and lists the files (`run.provenance.max_untracked_mb`). Otherwise a stray dump would land
  permanently in `.git/objects`.
- **Refusal** happens only when a snapshot cannot be made: not a git checkout, no HEAD, or the
  update-ref fails. In strict mode (`run.provenance=strict`) a dirty tree is also refused.
- **Immutability.** The `""` old-value makes creation atomic and fails if the ref exists. Refs are
  pushed and fetched without `+`, so a remote with a different sha for the same id rejects the
  update. Immutability is enforced by git itself.
- **Safe from gc.** Snapshot objects are reachable from `refs/runs/*`, so `git gc` keeps them.
  Deleting a ref is the only way to drop a snapshot.
- **Visibility.** The refs are absent from `git branch` and `git log`. They do show under
  `git log --all`, `gitk --all` and `git for-each-ref`. To hide them there:
  `git config log.excludeDecoration refs/runs/` plus the alias
  `git config alias.lga "log --exclude=refs/runs/* --all"`. I can't make `--all` skip them
  entirely; see the open questions.
- **Sharing.** `git push <remote> 'refs/runs/*:refs/runs/*'` and
  `git fetch <remote> 'refs/runs/*:refs/runs/*'`, both done by `xp sync` (§9).

**Borrowed.**
- DVC experiments do exactly this: `dvc exp run` commits the workspace into hidden
  `refs/exps/...`, shared by `dvc exp push/pull`.
- W&B instead stores `git rev-parse HEAD` plus a `diff.patch`. A patch is fragile: untracked files
  are missed, and it needs the base to apply. A commit is exact.
- Sacred records `dirty: true` and stops there. We keep the bytes.

## 3. The ledger: one JSON file per run

Path: `experiments/ledger/<host>/<run_id>.json`. It is written by the run, and only by the run.

**Writes are atomic:** write `<file>.tmp`, then `os.replace`. They happen at four points:
1. **start:** after the snapshot, before the run directory or any step;
2. **every checkpoint:** `updated`, `last_step`, `checkpoints`;
3. **the end:** on normal exit, on exception (status `failed` with the error), and on SIGTERM
   (`terminated`; Slurm sends SIGTERM before SIGKILL);
4. **never by anything else.**

A run killed by SIGKILL stays `running`. `xp ls` derives the label "lost" from an old `updated` time
plus sacct, but never writes it.

**Where the file goes.** It goes in the MAIN worktree's `experiments/ledger/`, found as
`dirname(git rev-parse --git-common-dir)` plus `experiments/ledger`, so runs from per-run
worktrees (§4) land in one place. It can be overridden with `XP_LEDGER_DIR`.

**Jobs never commit.** `xp sync` commits, in an ordinary git step (§9).

### Schema, version 1 (exact)

```jsonc
{
  "schema": 1,
  "run_id": "20261016T201503Z-torch-3fa9c1",
  "host": "torch",                                  // logical host; equals the ledger directory
  "status": "running",                              // running | completed | failed | terminated
  "created": "2026-10-16T20:15:03Z",                // ISO 8601 UTC, all times
  "updated": "2026-10-16T20:15:03Z",                // heartbeat (each checkpoint)
  "ended": null,
  "code": {
    "snapshot": "<40-hex>",                         // refs/runs/<run_id> points here
    "ref": "refs/runs/20261016T201503Z-torch-3fa9c1",
    "base": "<40-hex>",                             // HEAD at start
    "base_branch": "master",                        // symbolic HEAD, or null if detached
    "dirty": false,
    "dirty_paths": [],                              // first 50 changed or untracked paths
    "dirty_count": 0,
    "worktree": "/scratch/ax2348/mapping-wt/s-1a2b3c" // where it ran
  },
  "entry": {"argv": ["run.py", "+experiment=se2d_scaling_wsd", "model=se2d/hybrid4"],
            "cwd": "/scratch/ax2348/mapping-wt/s-1a2b3c", "via": "xp submit"},  // or "run.py", "xp run"
  "config": {
    "resolved": { /* the full resolved tree, as config.yaml */ },
    "overrides": ["+experiment=se2d_scaling_wsd", "model=se2d/hybrid4", "model.size=s"],
    "experiment_file": "conf/experiment/se2d_scaling_wsd.yaml",   // or an --exp path, or null
    "experiment_text": "# @package _global_\n...",               // the file's bytes, if --exp
    "sha256": "<of the resolved YAML>"
  },
  "run": {"task": "se2d", "model": "hybrid4", "size": "s", "name": "seed0_400k_qt_wsd",
          "seed": 0, "steps": 400000, "tags": ["major"]},          // tags from run.tag at launch
  "results": {"dir": "/scratch/ax2348/mapping/results/se2d/hybrid4/s/seed0_400k_qt_wsd",
              "root": "/scratch/ax2348/mapping/results",
              "rel": "se2d/hybrid4/s/seed0_400k_qt_wsd"},         // the RESOLVED path, decision (A)
  "parent": {"branch_from": null, "parent_run_id": null},          // --branch checkpoint and its run
  "reproduces": null,                                              // decision (B)
  "env": {
    "python": "3.12.14", "torch": "2.x", "cuda": "12.x", "cudnn": 91000,
    "fla": "0.x", "hydra": "1.3.5", "omegaconf": "2.3.1", "numpy": "2.x",
    "pip_freeze_sha256": "<hex>",                   // the full freeze goes in <results dir>/pip_freeze.txt
    "executable": "/scratch/ax2348/venvs/mapping/bin/python", "platform": "Linux-...",
    "node": "gh108"
  },
  "data": {"data/mnist.npz": {"sha256": "<hex>", "bytes": 12345678}},  // every bank or texture file opened
  "hardware": {"gpu": "NVIDIA H200", "gpu_count": 1, "driver": "5xx.xx"},
  "slurm": {"job_id": "20072852_4", "array_job_id": "20072852", "array_task_id": 4,
            "job_name": "map-h1c", "partition": "h200_public",
            "packed_with": ["<other run_id on this GPU>"]},       // or null for local runs
  "submit": {"id": "s-1a2b3c", "script": "jobs/h1c/s-1a2b3c.sbatch", "script_sha256": "<hex>"},  // or null
  "progress": {"last_step": 0, "checkpoints": []},
  "end": null
  // at the end:
  // "end": {"exit_status": 0, "reason": "completed", "error": null, "elapsed_s": 14317.2,
  //         "final_metrics": {"mse": 0.00073, "mse_last": 0.00124, "shuffled_mse": 0.1338,
  //                           "floor_mse": 0.0986, "best_step": 400000, "best_mse": 0.00073}}
}
```

- **Data hashes [mine].** Hashing each bank file once per run takes a few seconds for these sizes.
  It closes the gap that `data/` is gitignored. DVC's content addressing does the same job.
- **Model.** The brief's field list, laid out like Sacred's `run.json` (`experiment`, `host`,
  `dependencies`, `git`, `status`, `result`). Its pip-freeze hash stands in for Sacred's
  dependency list.
- **Why one file per run.** One writer per file means git merges cannot conflict. MLflow's file
  store uses the same layout, one directory per run. A single file with `merge=union` is the
  rejected alternative: union merge silently duplicates or interleaves JSON lines, and it still has
  two writers.
- **Checkpoints embed `run_id` and `code.snapshot` [mine].** So a `.pt` found anywhere leads back
  to its ledger entry (§8).
- **Size.** About 4-8 KB per run, mostly the resolved config. 1,000 runs is about 6 MB in git, and
  the files compress well.

## 4. Run directories: the template (decision A, confirmed)

Today `train.py:292` builds `os.path.join(task['root'], model.name, model.size, run_name)`. The
root is hard-coded per task in `mapping/tasks.py`: `results` for z2, `results/se2d`,
`results/se2c`, and the alias `se2` -> `results/se2`. The only escape is `run.out`.

The decided design:

```yaml
run:
  root: results
  dir: ${run.root}/${task.name}/${model.name}/${model.size}/${run.name}
  out: null          # the explicit override; wins when set
```

The Run object consumes both. **One flag on "must reproduce today's paths exactly":** z2 runs live
at `results/<model>/...`, with no `z2/` segment. The template as written would give
`results/z2/<model>/...` and break equivalence for z2. There are two ways to keep it exact.
- **(a) A per-task segment.** Add `task.results_subdir` (z2 `""`, se2d `se2d`, se2c `se2c`) to
  `conf/task/*.yaml` and use `dir: ${path_join:${run.root},${task.results_subdir},...}` with a
  resolver that drops empty segments.
- **(b) Override `run.dir` in `conf/task/z2.yaml`** to `${run.root}/${model.name}/...`.

I recommend (a): one template, data in the task file. It is my addition, so it is listed as open
question 1. The legacy alias stays a translator concern: `train.py --task se2` sets
`task.results_subdir=se2`.

- **What it buys.**
  - The ledger records the resolved path.
  - Tools look runs up in the ledger (`xp ls`, `mapping.provenance.find_runs`) instead of
    re-deriving paths, which retires `ALIASES` over time.
  - A preset may restructure: `run.dir: ${run.root}/campaigns/h1c/${model.name}/...`.
- **Each run directory gets a `RUN_ID` file [mine].** One line, written at creation. `xp sync`
  refuses to rsync a remote run into a local directory whose `RUN_ID` differs. That closes the m
  bridge collision class (audit, critical gap 3).
- **Results root on Torch.** With per-submit worktrees (§6), `run.root` must point outside the
  worktree, to a shared `/scratch/ax2348/mapping/results`. `xp submit` passes
  `run.root=<shared root>` as an override, so the config records it.

## 5. `run.py --reproduce` (decision B, confirmed)

`python run.py --reproduce <run_id> [overrides...] [--tolerance 1e-6]`:

1. Find `experiments/ledger/*/<run_id>.json`, after `git fetch <remote> refs/runs/<id>` and a ledger
   sync if needed. If it is missing, run `xp sync` first.
2. Check that the code object exists: `git cat-file -e <snapshot>^{commit}`. If not,
   `git fetch torch refs/runs/<id>:refs/runs/<id>`.
3. Create the worktree:
   `git worktree add --detach $TMP/xp-repro-<newid> <snapshot>`, then link `data/` into it. Check the
   ledger's `data` sha256s and refuse on a mismatch, which `--allow-data-drift` overrides.
4. In the worktree, write `experiment.yaml` (from `experiment_text` or the snapshot's
   `conf/experiment/...`) and `overrides.yaml`. Run THAT worktree's
   `python run.py --from <tmp>/config.yaml run.out=<out> <overrides>`, where `<out>` is
   `<original results dir>/reproductions/<UTC yyyymmddTHHMMSSZ>_<new run_id>/`.
   - `run.out` is the plain override, as decided.
   - The non-empty check only looks at this fresh folder.
   - Old snapshots predate `--reproduce`, but every Hydra-era snapshot has `--from`, which is all
     this step needs.
5. The new run makes its own snapshot. The worktree is clean at `<snapshot>`, so by §2 its snapshot
   equals the original's. Its ledger entry carries `"reproduces": "<original run_id>"`.
6. On exit, write `reproduce_report.json` into the reproduction folder and print the verdict:
   ```json
   {"original": "<id>", "reproduction": "<new id>",
    "verdict": "bitwise | within_tolerance | differs",
    "tolerance": 1e-6,
    "metrics": {"mse": {"original": 0.00073, "reproduction": 0.00073, "abs_diff": 0.0, "rel_diff": 0.0}},
    "checkpoint_bitwise": true,
    "code": {"original": "<sha>", "reproduction": "<sha>", "same": true},
    "env_diff": {"torch": ["2.5.1", "2.6.0"]},
    "data_diff": {},
    "config_diff": []}
   ```
   The verdict is `bitwise` when the final checkpoint tensors are identical.
7. Remove the worktree (`git worktree remove`), unless `--keep`.

**If the original is not local** (a Torch run reproduced locally), `<original results dir>` does not
exist. A stub parent is created: `results/.../<run>/` holding `LEDGER.json` (a copy of the ledger
record) and, with `--fetch-summary`, `final.json` and `log.jsonl` rsynced from the original host.
The stub has a `STUB` marker so it is never mistaken for the run.

**Aggregation must skip `reproductions/`:**
- `mapping/results.py` (the glob behind summarize.py and the fits);
- `scaling.py`'s fit readers;
- `tools/compare_grid.py` when given a directory;
- any `glob('results/**/final.json')`.

A shared helper `mapping.provenance.iter_run_dirs(root)` does the skipping, and a test plants a
`reproductions/` child and asserts summarize.py ignores it.

**Pre-Hydra runs** (136 legacy) have no snapshot. `--reproduce` on a retro-imported id (§11) is
refused. `--reproduce-approx` builds the worktree at `code.inferred` and runs the frozen
`train.py` flags from `args`. Its report is labelled `approximate (inferred commit)`.

## 6. Torch trees become real git clones (migration plan; read-only today)

Today there are four rsync exports. Target:

```
/scratch/ax2348/mapping.git          bare hub; local remote "torch" = torch:/scratch/ax2348/mapping.git
/scratch/ax2348/mapping              main clone of the hub (branch master); experiments/ledger/torch/ is written here;
                                     data/ -> /scratch/ax2348/mapping-data (shared, read-only)
/scratch/ax2348/mapping-wt/<submit>  one detached worktree per submission, at its snapshot (immutable code for queued jobs)
/scratch/ax2348/mapping/results      the shared run.root for every Torch run (new runs)
```

- **Why a bare hub.** Torch nodes cannot reach the laptop, and pushing into a non-bare checked-out
  branch is refused (`receive.denyCurrentBranch`). Local pushes to the hub, and the main clone
  fetches from it on the same filesystem.
- **Why per-submit worktrees [mine].** The audit's worst Torch finding is that a push changed the
  code under queued jobs. With worktrees, nothing ever changes a submitted job's code. A worktree
  is about 3 MB of code. `xp sync --prune` removes worktrees whose jobs are no longer
  queued or running.

**Steps.** Nothing below runs until the user approves, and none of it touches the existing trees.

1. **Archive the exact bytes of each old tree first [mine].**
   - For each of `mapping`, `mapping_lift`, `mapping_hydra` and `mapping_odq`, rsync its code
     (excluding results, data, logs and `*.pt`) to a local temporary directory.
   - Commit it against the closest commit:
     `GIT_INDEX_FILE=... GIT_WORK_TREE=<dir> git add -A && git write-tree && git commit-tree -p <closest commit from the audit>`.
   - Store it as `refs/archive/torch/<tree>/2026-10-16`.

   This preserves, for example, `mapping_lift`'s five blind-probe files that differ from 038f19c,
   and gives retro-imported runs (§11) an exact "code currently on disk" commit to cite.
2. Push exp/odq and every other branch to the hub, so a353d1e is no longer only in
   `~/code/mapping-hydra` (audit, high gap 5).
3. Set up the hub and clone:
   - `ssh torch git init --bare /scratch/ax2348/mapping.git`;
   - locally: `git remote add torch torch:/scratch/ax2348/mapping.git`, then
     `git push torch --all && git push torch --tags && git push torch 'refs/archive/*:refs/archive/*'`;
   - on Torch: `git clone /scratch/ax2348/mapping.git /scratch/ax2348/mapping-git`, under a NEW
     name; then `ln -s /scratch/ax2348/mapping-data data`.
   - Verify the clone with `git status --porcelain` (expect empty) and the audit's blob-hash match
     against the old trees.
4. Leave the old trees untouched until every job that `cd`s into them has finished: H1c/H1d in
   `mapping_hydra` and odq in `mapping_odq`. Check with `squeue -u ax2348 -o %Z`.
   - Then rename them to `/scratch/ax2348/archive/<tree>` and write a `FROZEN` note in each.
   - Their `results/` stay where they are. The retro ledger (§11) records those absolute paths.
5. Swap the names: `mapping-git` becomes `mapping` (after the old `mapping` is archived). Point the
   venv and `~/torch-gpu`'s `--dir` default at it.
6. `scripts/sync.sh` is retired in favour of `xp sync` (§9). Its `push` mode, which is the rsync
   that caused the problem, is removed. `pull` survives inside `xp sync --results`, with the
   `RUN_ID` guard.

## 7. Job templates

There is one template: `jobs/templates/torch.sbatch` (`string.Template`, `$`-placeholders). It
replaces the hand-written ones.

```bash
#!/bin/bash
#SBATCH --job-name=map-$campaign
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=$time
#SBATCH --array=0-$last%$concurrency
#SBATCH --output=/scratch/ax2348/mapping/results/jobs/logs/$campaign-%A_%a.out
set -uo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping-wt/$submit_id                 # immutable code
export XP_HOST=torch XP_SUBMIT_ID=$submit_id XP_LEDGER_DIR=/scratch/ax2348/mapping/experiments/ledger
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs/$campaign/$submit_id.tasks)
# one line = one GPU; " | " separates runs packed onto it; each run is a run.py argument list
IFS='|' read -ra runs <<< "$line"; pids=()
for r in "${runs[@]}"; do eval "python run.py $r run.root=/scratch/ax2348/mapping/results" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
```

- The task file holds plain `run.py` override lists, so the sbatch carries no model logic.
- Packing is written explicitly in the task file. The runs on one GPU learn each other's ids from
  `XP_PACK_GROUP` and record them in `slurm.packed_with`.
- `scaling.py --slurm` and `tools/branch_jobs.py` render through this template, retiring their own
  copies.

## 8. Enforcement (tests)

These extend `tests/test_seal.py` and add `tests/test_provenance.py`. Each one runs on a temporary
git repo (a `git init` fixture with a copy of the harness), on CPU, with a 2-step run.

1. **Single entry.** The existing `test_only_run_py_trains` (AST scan) stays.
   - It is extended to `experiments/` and `mapping/xp/`. `xp` must never call `train.main`; it
     invokes `run.py` as a subprocess.
   - The blind probe's `experiments/lift_blind_probe/train.py` is either exempted by name with a
     reason, or ported (open question 5).
2. **Provenance before training.**
   - `train.main` calls `provenance.begin(...)`, which makes the snapshot and the ledger, after the
     existing "everything is checked" point and before `os.makedirs(out)`.
   - Test: monkeypatch `begin` to raise. Assert that no run directory exists and that the step
     counter never moved.
3. **No checkpoint without a ledger.**
   - Checkpoints are written only through `mapping.checkpoint.save(ledger, ...)`, which needs a live
     ledger handle and embeds `run_id` and `snapshot`.
   - An AST test asserts that `torch.save` appears nowhere in the training path except in that
     function. Today `train.py:488` calls `torch.save` directly, beside an unused
     `mapping.checkpoint.save_checkpoint`.
   - Test: a 2-step run with `save_every=1`. Every `*.pt` carries a `run_id` whose ledger file
     exists and whose `code.snapshot` resolves (`git cat-file -e`).
4. **Snapshot hygiene.**
   - HEAD, `git symbolic-ref HEAD`, the index's sha1 and `git status --porcelain` are unchanged
     after a run.
   - An untracked file is in the snapshot tree; a `.gitignore`d one is not.
   - A file written under `experiments/ledger/` leaves `dirty == false`.
   - A clean tree gives `snapshot == HEAD`.
   - Creating the same ref twice fails.
5. **Refusals.** Outside a git checkout, `run.py` exits non-zero before writing anything. With
   `run.provenance=strict` and a dirty tree, it exits non-zero. An untracked file over the size
   guard makes it refuse and name the file.
6. **Ledger shape.** The start record validates against the schema in §3, which is kept as
   `tests/ledger_schema.json` (stdlib checker, no new dependency). After the end record, `status`
   is `completed` and `end.final_metrics.mse` equals `final.json`. SIGTERM mid-run gives
   `terminated`.
7. **Job templates.** Every `jobs/**/*.sbatch` added after the cutoff commit, and every generator's
   output, runs `python run.py` from a `mapping-wt/` directory. The cutoff exempts the records
   already in `jobs/lift/`.
8. **Equivalence.** The run-directory template (§4) reproduces today's paths for z2, se2d, se2c and
   the se2 alias. The existing bitwise behaviour tests stay green, since provenance adds no
   randomness.

## 9. Tooling: `xp`, porcelain over the plumbing

People and agents should not have to retype git, ssh or sbatch incantations. `xp` is a thin CLI
over plain files and git.

**Principles (the user's):**
- It WRAPS and does not hide: `--explain` prints every underlying command before running it, and
  `--dry-run` prints without running.
- The data is plain text (ledger JSON, refs, registry YAML). Losing `xp` loses convenience, never
  data.
- Query commands take `--json` for agents.
- Commands are idempotent and safe: re-running `sync` or `submit` never duplicates.
- `run.py` stays the only training entry. `xp` shells out to it, so the seal tests hold.

**Name.** `xp` is short and free on both machines (`command -v xp` finds nothing locally, on the Torch login node, or in the Torch venv). I see no reason to
change it.

**Packaging [mine].**
- **Code.** It lives at `mapping/xp/` (`cli.py`, `git.py`, `remote.py`, `ledger_query.py`). It
  imports only `mapping.provenance` (snapshot, ledger I/O, run lookup, `iter_run_dirs`) and the
  standard library. `run.py` uses the same `mapping.provenance`.
- **pyproject.toml.** There is none today. A minimal one would make `mapping` installable:
  `[project] name="mapping"`, `[project.scripts] xp = "mapping.xp.cli:main"`, setuptools package
  discovery. Install it with `pip install -e .` in the local conda env and the Torch venv.
- **No new runtime dependency.** argparse, subprocess and json only; YAML through the existing
  omegaconf.

**Configuration.** `conf/hosts.yaml` holds:
- `torch: {ssh: torch, hub: /scratch/ax2348/mapping.git, clone: /scratch/ax2348/mapping, worktrees: /scratch/ax2348/mapping-wt, results: /scratch/ax2348/mapping/results, venv: /scratch/ax2348/venvs/mapping}`;
- `erid-2: {gpu_lock: <path>}`.

### Subcommands and exactly what they run

**`xp run <preset|--exp FILE> [overrides] [--tag major] [--bg]`**: a local run.
- It takes `flock <gpu_lock>` for the whole run. This is the rule of one sequential launcher per
  GPU: a second `xp run` waits; it does not start a second conditional launcher.
- It runs `python run.py +experiment=<preset> <overrides> run.tag=[major]`.
- `--bg` adds `nohup ... > results/jobs/logs/local-<UTC>.out 2>&1 &` and prints the run id read
  from the log's first line.
- The snapshot and ledger are made by run.py, not by xp.

**`xp submit <preset|grid.yaml> --on torch [--concurrency N] [--pack 2] [--time 16:00:00] [--resubmit]`**:
- **(1) Local preflight.**
  - Expand the grid into run.py override lines: a grid YAML lists a preset plus axes.
  - Snapshot the local tree, with the same mechanism as §2, to `refs/submits/<submit_id>`.
  - `submit_id = s-<first 6 hex of sha256(snapshot sha + task file + campaign)>`. This hash is the
    idempotency key.
- **(2) Idempotency.**
  - If `experiments/ledger/submits/<submit_id>.json` exists and sacct shows its job PENDING,
    RUNNING or COMPLETED, refuse: "already submitted as <job>".
  - `--resubmit` salts the id instead.
- **(3) Push:** `git push torch <snap>:refs/submits/<submit_id>`, never with `+`.
- **(4) Remote.** Over `ssh torch` (one session, `set -e`):
  - `git -C <clone> status --porcelain -- . ':!experiments/ledger'` must be empty. Otherwise
    refuse: "the Torch clone has local edits".
  - `git -C <clone> fetch origin refs/submits/<id>:refs/submits/<id>`
  - `git -C <clone> worktree add --detach <worktrees>/<id> <snap>`
  - `ln -s /scratch/ax2348/mapping-data <worktrees>/<id>/data`
  - `test -x <venv>/bin/python && <venv>/bin/python -c 'import hydra, omegaconf, fla'`
- **(5) Write and copy the job files.**
  - Render the template (§7) to `jobs/<campaign>/<submit_id>.sbatch` and `.tasks` locally. They are
    tracked as the record.
  - `scp` both into the worktree's `jobs/<campaign>/`. They are the same bytes; the sha256 is
    checked after the copy.
- **(6) Submit:** `ssh torch sbatch --parsable <worktrees>/<id>/jobs/<campaign>/<id>.sbatch` gives
  the job id.
- **(7) Record.**
  - Write `experiments/ledger/submits/<submit_id>.json` (a single writer: this local submit).
  - Fields: `{submit_id, snapshot, campaign, preset, task_lines, sbatch_sha256, job_id, submitted}`.
  - Runs started by the job write their own ledger files with `submit.id`.
- **Not "behind" but immutable.** With per-submit worktrees the Torch clone cannot be behind for a
  job, since the job runs the pushed snapshot. The brief's "refuse if the clone is behind" becomes:
  refuse if the push is rejected, or if the clone has local edits.

**`xp ls [--running] [--tag T] [--model M] [--task T] [--host H] [--since 2d] [--submit ID] [--fetch] [--json]`**:
- It reads `experiments/ledger/*/*.json`. `--fetch` runs `xp sync --pull-only` first.
- `--running` shows `status == running` with `updated` within 2x the checkpoint interval. Older
  ones show as `lost?`.
- `--check` adds `ssh torch sacct -j <ids> -X -n -P -o JobID,State`. Nothing is written.
- Submits with no run file yet show as `submitted (no run yet: <sacct state>)`.

**`xp show <run_id prefix>`**:
- the ledger;
- `git show --stat <snapshot>`;
- if dirty, `git diff --stat <base> <snapshot>`;
- the results directory, a `ls` of its checkpoints, the final metrics, and the registry entry if
  any.

**`xp compare <ids...> [--metric mse,mse_last] [--grid]`**:
- a metrics table from `end.final_metrics`;
- a config diff (`run.py`'s `differences()` on `config.resolved`);
- a code diff, `git diff --stat <snapA> <snapB>`;
- `--grid` runs `python tools/compare_grid.py <dirA> <dirB> --labels ...` on the resolved results
  directories.

**`xp reproduce <id> [--on torch]`**:
- locally: `python run.py --reproduce <id>` (§5);
- with `--on torch`: `xp submit` of a one-line task `--reproduce <id>`, which reuses the original
  snapshot (no new push needed, since `refs/runs/<id>` is already on the hub).

**`xp sync [--results] [--prune] [--pull-only]`**:
- It takes a lock with `flock .git/xp-sync.lock`.
- **1. Push local refs:**
  `git push torch 'refs/runs/*:refs/runs/*' 'refs/submits/*:refs/submits/*'`.
- **2. Commit Torch's ledger on Torch, without touching its checkout.** Over ssh, in the clone:
  ```sh
  export GIT_INDEX_FILE=$(git rev-parse --git-dir)/xp-ledger-index
  git read-tree ledger/torch 2>/dev/null || git read-tree --empty
  git add experiments/ledger/torch
  t=$(git write-tree); p=$(git rev-parse -q --verify ledger/torch)
  [ "$t" = "$(git rev-parse ledger/torch^{tree} 2>/dev/null)" ] || \
    git update-ref refs/heads/ledger/torch $(git commit-tree $t ${p:+-p $p} -m "ledger: torch") $p
  git push origin ledger/torch 'refs/runs/*:refs/runs/*'
  ```
  This is the same temporary-index trick as §2. The `$p` old-value makes the update compare-and-swap.
  Only Torch writes this branch.
- **3. Fetch:**
  `git fetch torch 'refs/runs/*:refs/runs/*' ledger/torch:refs/remotes/torch/ledger/torch`.
- **4. One tidy local commit.**
  - `git checkout refs/remotes/torch/ledger/torch -- experiments/ledger/torch`, then
    `git add experiments/ledger/<localhost> experiments/ledger/submits`.
  - If `git diff --cached --quiet -- experiments/ledger`, do nothing: that is the idempotence.
  - Otherwise `git commit -m "ledger: sync (torch +N ~M, erid-2 +K)" -- experiments/ledger`. It is
    explicit-path, so other staged work is never swept in.
- **5. `--results`.** Rsync each new or updated Torch run's results directory into the local results
  root at its ledger `results.rel`. Before copying, compare the destination's `RUN_ID`. On a
  mismatch, refuse and name both ids.
- **6. `--prune`.** Remove the worktrees of finished submits (remote `git worktree remove`), keeping
  `refs/submits/*`.
- **No conflicts by construction.** Each host directory has one writer, and `submits/` is written
  only locally.

**`xp tag <id> major [--note TEXT]`.**
- The ledger is single-writer (the run), so `xp tag` does NOT edit it. It appends the id, the note
  and the date under `tagged:` in `experiments/registry.yaml`, which is the human-curated file.
- Launch-time tags go in the ledger (`run.tag=major`).
- `xp ls --tag major` unions the two.

**`xp gpu <args>`.**
- A passthrough to `~/torch-gpu`, with `--explain` support.
- `xp gpu run <preset> [overrides]` sends `cd <clone> && python run.py ...` into the held session
  (`torch-gpu send`). Runs on a held GPU therefore still get a snapshot and a ledger file.
- The held session runs from the main clone, not a worktree. That clone is protected by the
  local-edit check, and its runs snapshot its exact state.

### Failure modes

| failure | behaviour |
| --- | --- |
| ssh to torch fails | `submit` and `sync` stop before any remote change. Local commits are not made. Exit 3 with the ssh error. |
| push rejected (ref exists with a different sha) | Abort as an immutability violation. Print both shas. Never retried with force. |
| Torch clone has local edits | `submit` refuses and lists them. Nothing is pushed or submitted. |
| sbatch fails after the worktree was created | The worktree is left in place and the submits record gets `job_id: null, error`. A rerun reuses the same worktree, by idempotent id. |
| job dies before run.py starts (venv, node) | No run ledger file. `xp ls` shows the submit as "no run yet" with the sacct state (FAILED). |
| run SIGKILLed or node lost | The ledger stays `running`. `xp ls` shows `lost?` once `updated` is stale, and `--check` confirms through sacct. Nothing is rewritten. |
| corrupt or partial ledger file | Impossible by atomic replace; if found, `xp ls` warns and skips it. |
| two `xp sync` at once | `flock` serialises them, and the remote update-ref is CAS. |
| results rsync into another run's directory | Refused on `RUN_ID` mismatch (closes the collision class). |
| scratch quota from worktrees | `--prune`. `xp submit` checks `df` and refuses below 1 GB free. |
| a large untracked file in the snapshot | `run.py` refuses with the list (§2 size guard). |
| clock skew between hosts | Ids carry a host part and random hex, so uniqueness never depends on clocks. |

### Test plan for xp

- **Fixtures.**
  - A temporary local repo plus a temporary bare "hub".
  - A fake `ssh`: `XP_SSH` points at a script that runs the command locally, with a cwd mapped to a
    fake Torch root.
  - Fake `sbatch`, `sacct` and `squeue` scripts on `PATH`, recording their calls and returning
    canned ids and states.
  - Runs are 2-step CPU runs.
- **Idempotence.**
  - `xp sync` twice gives exactly one new commit, and the second run reports "nothing to sync".
  - `xp submit` twice gives one sbatch call, and the second refuses with the job id.
  - `--resubmit` gives a new id.
- **Safety.**
  - A pre-existing `refs/runs/X` with a different sha on the hub makes `sync` fail without force.
  - A dirty fake Torch clone makes `submit` refuse.
  - A `RUN_ID` mismatch makes `--results` refuse.
- **`--explain`.** A golden test of the printed command list for `submit` and `sync`.
- **`--json`.** `xp ls --json` validates and round-trips the ledger fields.
- **Seal.** The AST scan covers `mapping/xp/`: no `train.main`, no `torch` import. `xp run` spawns
  `run.py` (asserted through the fake process list).
- **End to end (fixture).**
  `xp submit` -> the fake sbatch runs the template inline -> `run.py` writes the ledger into the
  fake clone -> `xp sync` -> the local repo has the ledger commit and `refs/runs/<id>` -> `xp show`
  and `xp reproduce` (CPU, bitwise) report `bitwise`.

### Compared with DVC and W&B

| operation | DVC experiments | W&B CLI | xp |
| --- | --- | --- | --- |
| launch | `dvc exp run` snapshots the workspace to `refs/exps/...` and runs the `dvc.yaml` stage | `wandb launch` / agents (needs the server) | `xp run` / `run.py`: snapshot to `refs/runs/<id>`, no pipeline file |
| list | `dvc exp show` (params and metrics table from git refs) | `wandb` web UI, API | `xp ls` / `xp compare` from ledger JSON |
| share | `dvc exp push/pull <remote>` (git refs) | `wandb sync` uploads offline runs | `xp sync` (git refs + a ledger branch) |
| remote submit | none built in (`dvc queue` runs locally) | `wandb launch` to a queue | `xp submit` (ssh + sbatch, per-submit worktree) |
| code capture | a workspace commit | commit sha + `diff.patch` | a workspace commit (the DVC model) |

- **What we take from DVC:** hidden refs and push/pull of refs. We skip its pipeline DSL and cache.
- **What we take from W&B:** offline-first logging, then `sync`, and the metadata it captures (git,
  pip freeze, host, GPU). We skip the server, as the brief says.
- **MLflow:** a later `xp export --mlflow <dir>` could write the file-store layout (`mlruns/0/<run>/`
  `meta.yaml`, `params/`, `metrics/`, `tags/mlflow.source.git.commit`) from the ledger. It is not
  in scope.

## 10. Major going forward, and the registry

- **Every run is in the ledger.** "Major" is either `run.tag=major` at launch (in the ledger) or
  curated later in `experiments/registry.yaml`, through `xp tag` or by hand.
- **Registry entries move from paths to ledger ids** once runs have them. The retro-imported ids
  (§11) let the existing 13 major experiments switch now.
- **Reports [mine, open].** A published report should carry the list of run ids it was built from,
  plus the git sha of `experiments/ledger` at build time, in its footer or data block. It should
  not use a "data pulled HH:MM" stamp over an untracked copy.

## 11. Existing runs, retroactively

`xp import-legacy experiments/audit/runs.csv` writes one ledger file per REAL run (159; smoke rows
skipped) to `experiments/ledger/<host>/legacy-<host>-<task>-<model>-<size>-<run>.json`. `#local`
becomes `-local`. Each file has `"schema": 1, "provenance": "retro"`, and the code block is:

```json
"code": {"snapshot": null, "ref": null,
         "recorded": "8837c28",                     // from a LIFT_COMMIT echo, or null
         "inferred": "7c1f827",                     // the audit's time-based inference
         "basis": "last master commit before submit; rsync export, dirty unknown",
         "archive": "refs/archive/torch/mapping_lift/2026-10-16",   // §6 step 1, when the tree still matches
         "tree_verified": "2026-10-16: mapping_hydra byte-identical to 50341e4",  // only where proven
         "dirty": null}
```

- **Other fields.** `config.resolved` comes from `config.yaml` for the 17 Hydra runs. For legacy
  runs it comes from `args`, translated by the frozen `train.py` (`overrides_from_flags`) and
  marked `"config_source": "legacy-flags"`. Slurm ids, node and GPU time come from the audit
  (sacct). `results.dir` is the run's absolute path on its host. `status` and `final_metrics` come
  from `final.json`.
- **No `refs/runs/` for retro runs.** A ref would claim an exactness that does not exist. The
  `refs/archive/torch/*` commits are cited instead.
- **Imported, not created.** Retro files are written once by the importer, which is their single
  writer, and are never touched by runs. The 10 runs still running at import time are imported as
  `running` and re-imported once (`--refresh`) when they finish. They are pre-provenance jobs and
  will never write a ledger themselves.
- **Before any import.** Settle the m bridge collision (rename the local run's directory, on the
  user's yes) and fix the STATUS seed-1 line. Both are in the audit.

## 12. Code-structure audit (decision C): proposal only, no refactoring

### Current layout, as it stands at 380d929

| place | what is there | assessment |
| --- | --- | --- |
| `train.py` (603 lines, root) | the trainer `main(experiment)` AND the frozen argparse CLI (`parse`, `flags`, `overrides_from_flags`, `main_from_flags`) AND `write_sources` | Two responsibilities. The trainer is library code; the CLI is a deprecated translator. |
| `run.py` (267, root) | the Hydra compose, rerun and print entry | Right place. Its helpers `load`, `resolved`, `differences` and `merge_overrides` are imported by train.py and tools, so they are library code living in an entry script. |
| `render.py`, `summarize.py`, `plot_curves.py`, `profile_loader.py`, `scaling.py` (root) | thin CLIs over `mapping/render.py`, `mapping/results.py`, `mapping/plot.py`; a loader profiler; the sweep and fit generator | The CLIs are fine as entry points, but five root scripts with tools/ beside them is two homes for the same kind of thing. `profile_loader.py` belongs in tools/. scaling.py's job-writing duplicates `tools/branch_jobs.py` (two sbatch templates in strings). |
| `mapping/task.py`, `task_se2d.py`, `task_se2c.py`, `se2d.py`, `se2c/`, `tasks.py`, `data.py` | z2's task is unsuffixed `task.py`; se2d's geometry is `se2d.py` beside `task_se2d.py`; se2c is a package (`se2c/`) plus `task_se2c.py` | Three naming schemes for one concept. `data.py` (437) still holds the z2/se2d bank generator and `Config`, with the heading switch for se2d. |
| `mapping/models/*.py` vs `mapping/models/se2d/*.py` | the z2 models at the top level; se2d copies under `se2d/`. 10 pairs differ by only 32-46 lines each (for example `hybrid4.py` 597 vs `se2d/hybrid4.py` 619). The ablations are full ~600-line copies differing from se2d/hybrid4 by 20-50 lines. | The largest duplication: about 14k lines in `mapping/models/{,se2d/}*.py`, much of it repeated; I have not measured exactly how much is distinct. A mixer fix must be made in up to 10 files. This is deliberate history (frozen arms), but now a maintenance risk. |
| `mapping/checkpoint.py` | `save_checkpoint` / `load_checkpoint` | train.py does not use it (it calls `torch.save` at line 488): a duplicated helper. |
| `mapping/config.py` | the config objects (Experiment, Run, Task, generators, schedules) | Right place. The run-directory template (§4) goes here. |
| `mapping/tasks.py` | the task table, with hard-coded `root` per task and `ALIASES` | Roots move into config (§4). The alias shrinks to the translator. |
| `mapping/ocean/` | the ocean renderer (exp/ocean) | Fine as a subpackage. Its CLIs are in tools/. |
| `tools/` | probes (`probe_p5`, `transition_probe`, `read_beta`, `probe_gdn`, `time_scan_probe`), audits (`audit_se2d`), benches, sheets and videos, `compare_grid`, `branch_jobs`, `registrability` | Mixed: analysis CLIs, which is right, plus a job generator and probe math that other tools import. Shared probe code (loading a checkpoint, running the delta-rule scan) is repeated across the probe tools. |
| `scripts/` | data download (`get_*`), `sync.sh`, `remote_setup.sh`, `pyhdr.sh` | Fine. `sync.sh` is replaced by `xp sync`. |
| `experiments/` | `lift_blind_probe/` (its own harness and trainer); now `audit/`, `registry.yaml`, and later `ledger/` | Two meanings: a self-contained side harness, and provenance data. The side harness is a second training entry point. |
| `conf/`, `jobs/`, `tests/` | config tree; launch records; tests | Fine. `jobs/templates/` is new (§7). |
| `quarantine/`, `shelved/` | gitignored and old code | Fine as is. |

### Target layout

```
mapping/
  config.py            config objects; Run gains root/dir (§4)
  compose.py           from run.py: load, resolved, build, differences, merge_overrides, rerun
  train.py             the trainer: main(experiment, config, sources); from root train.py
  legacy_cli.py        the frozen flag translator: parse, flags, overrides_from_flags, main_from_flags
  provenance.py        snapshot, ledger I/O, find_run, iter_run_dirs (skips reproductions/), RUN_ID
  checkpoint.py        THE checkpoint writer (needs a ledger handle) and reader
  results.py           reading runs back (uses provenance.iter_run_dirs / the ledger)
  tasks/               __init__ (registry), z2.py, se2d.py (+ geometry), se2c/ (package as today)
  models/
    blocks/            shared mixers: attention+RoPE, Gated DeltaNet, SwiGLU, short conv
    z2/  se2d/  se2c/  per-task models as thin compositions of blocks
  xp/                  cli.py, git.py, remote.py, ledger_query.py   (console script `xp`)
  ocean/  render.py  plot.py  data.py
run.py                 3 lines: from mapping.compose import cli; cli()
train.py               3 lines: the frozen flags entry, printing the deprecation line, calling legacy_cli
summarize.py render.py plot_curves.py scaling.py   thin CLIs (unchanged names, so docs and habits hold)
tools/                 analysis CLIs only; shared probe code moves to mapping/probes.py
```

`xp` lives in `mapping/xp/`. It is a package module with the console-script entry, kept distinct
from training: it never imports `mapping.train` or torch.
- **What it imports:** `mapping.provenance` (snapshot, ledger, lookup), `mapping.compose` (for
  `differences`, used by `xp compare`) and `mapping.config` (path resolution only).
- **What `run.py` imports:** the same `provenance` and `compose`.

### Migration order (each step is a separate commit with the full suite green)

1. **Add `mapping/provenance.py` and `mapping/xp/`** (new code only), plus `pyproject.toml`. The
   suite is unchanged.
2. **Move run.py's helpers into `mapping/compose.py`.** `run.py` re-exports them, so imports by
   tools and tests keep working. The equivalence tests are unchanged.
3. **Split `train.py`.** The trainer goes to `mapping/train.py` and the CLI to
   `mapping/legacy_cli.py`. Root `train.py` stays a shim, keeping the frozen-flags snapshot test and
   the seal test, whose expected starter changes from `run.py` -> `train.main` to `run.py` ->
   `mapping.train.main`, stated in the test.
4. **Route checkpoints through `mapping/checkpoint.py`,** with a ledger handle and `run_id`
   embedded. The bitwise tests compare model tensors, not file bytes, so they stay green. The
   checkpoint *file* gains keys.
5. **Add the run-directory template** (§4), with a path-equivalence test per task and alias, then
   remove `root` from `tasks.py`.
6. **Rename the task modules** `task.py`, `task_se2d.py`, `se2d.py` and `task_se2c.py` into
   `mapping/tasks/...`, with import shims at the old names for one release. The contract tests are
   unchanged.
7. **Factor the model blocks, one model family at a time, each behind a bitwise forward test** (the
   same seed and weights give an identical output before and after; `tests/golden.py` already has
   the pattern).
   - This is the only risky step. Recommend doing it last, and only for actively developed models
     (hybrid4, vanilla4 and the causal, time and mix variants).
   - Frozen ablation arms (beta1, cdiag, noconv, gelu, hybrid45b*) stay as copies, marked frozen.
8. **`tools/`:** move shared probe code to `mapping/probes.py`, and `profile_loader.py` into tools/.
   `scaling.py` and `branch_jobs.py` render the single template (§7).

## 13. Open questions for the user

1. **The template and z2 paths (A).** The confirmed `${run.root}/${task.name}/...` would move z2 to
   `results/z2/...`. Should exactness come from (a), a `task.results_subdir` segment that is empty
   for z2 (recommended), or (b), a z2-specific `run.dir`?
2. **The ledger in the snapshot.** The brief excludes `experiments/ledger/` from the dirty check. I
   also exclude it from snapshot content. Agree?
3. **Untracked files.** Is the size guard at 20 MB right? Should the refusal be a warning instead?
4. **`git log --all`.** It shows `refs/runs/*`, and git offers no way to opt a namespace out of
   `--all`. Is `log.excludeDecoration` plus an alias enough?
5. **The blind-probe harness** (`experiments/lift_blind_probe/train.py`). Should the seal exempt it
   by name, or should it be ported to run.py? It is shelved, so I lean towards exempting it.
6. **Migration timing.** H1c/H1d (up to 16 h per task, with 25 of their array tasks pending, plus 3 odq) and odq keep the old
   trees busy. Is it OK to build the hub and new clone beside them now, and archive the old trees
   only when their queues drain?
7. **Off-machine backup.** `~/code/mapping` has no remote. Its only other copies are mapping-hydra
   (same disk) and the Torch exports. The Torch hub would be the first real second copy. Should
   there also be a private GitHub remote? `refs/runs` and the ledger would go there too.
8. **Retro import scope.** Import all 159 real runs, or only those in the major registry (147)?
9. **The audit scripts.** They sit in the session scratchpad. Should they be committed as
   `tools/provenance_audit.py`, or folded into `xp import-legacy`?
10. **Reports citing run ids** (§10): adopt it as a rule for future published pages?
11. **Two items from the audit that need a yes before anyone acts:**
    - rename the local `results/se2c/hybrid4c/m/bridge_n4dec_100k_qt` to `..._local` before the next
      pull;
    - correct STATUS: 20028847_0 is hybrid4 **s** seed 1, not xs.
12. **Strict mode default.** Off locally and implicit on Torch, since submits always come from a
    snapshot? Or on everywhere?
13. **Build vs adopt (§0b).** I recommend BUILD. Is the DVC hybrid (`dvc exp save -I` as the snapshot primitive) worth its dependency for an established ref format?
14. **The model-block refactor** (§12, step 7). Do it at all, or keep the per-model copies as frozen
    history?
