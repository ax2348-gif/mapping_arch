# Configuration (Hydra)

A run is configured by composing the tree in `conf/` and instantiating it
into the objects in `mapping/config.py`; the trainer (`train.main`) reads
every setting from those objects. The rule: **every field in a composed
config is consumed by the trainer**.
- Variants (which generator a task uses, which schedule, which size) are
  chosen by the defaults list, so a run never carries the fields of a
  variant it does not use.
- Every node instantiates a class with a strict signature, so an unknown
  or misspelled key is an error, not an ignored setting.

`tests/test_config.py` enforces this. A short real run of the trainer
must read every field of every object, and changing any field of every
model file, schedule and preset must change what the trainer trains on
(the generator config, the learning rate at each step, the model it
builds), or be refused.

Behaviour is unchanged from the `train.py --flags` harness:
`tests/test_equivalence_runs.py` trains short runs on the pre-refactor
code and on both entry points here, and compares everything they write.

## Launch

```
python run.py model=se2d/hybrid4 run.note=smoke             # task se2d, size xs, cosine schedule
python run.py model=se2d/hybrid4 model.size=m schedule=wsd run.steps=400000 \
    "run.note='m at 400k, WSD, for the efficiency curve'"
python run.py model=z2/vanilla model.size=l run.note=rung   # a rung of the model's own ladder
python run.py +experiment=se2c_bridge model.size=s run.note=bridge   # a preset (conf/experiment/)
python run.py --exp ~/notes/my_run.yaml run.seed=1 run.note=seed1    # your own experiment file
python run.py --from results/se2c/hybrid4c/s/bridge_n4dec_100k_qt/config.yaml run.name=again \
    schedule=wsd "run.note='bridge s again, WSD'"
python run.py ... --print                             # resolved config, and the exact train.py command if any; no run
```

- **The run note (required).** Every run needs `run.note`, one line saying
  what it is for, like `git commit -m`; one word is enough, empty is
  refused. It is given at launch only: no file under `conf/` sets it (a
  test checks), an experiment file or a saved run that sets it is refused,
  and `--from` never inherits it. Quote it for Hydra when it has spaces,
  commas or colons: `"run.note='a, b: c'"`. `xp run <preset> -m '<note>'`
  sets it; `train.py --flags` takes `--note '<note>'`. Job generators
  (scaling.py, tools/branch_jobs.py) write one per task, `xp reproduce`
  writes `reproduce <run_id>`. It is launch metadata: it is in
  `config.yaml`, `provenance.json` and the ledger, never in the `args`
  record or anything the trainer trains on (tests/test_provenance.py).
- **Tags.** `run.tag=[h1c]` (a list; a preset may set it) labels a
  campaign. Ledger only, like the note.
- **Arguments.** Hydra overrides: `key=value`, `group=option` and
  `+experiment=name`. A mistake (no model, an unknown model or field, an
  architecture override, a comma sweep, an experiment file without its
  header) is an error saying what to do, not a Hydra traceback.
- **`run.py` and `train.py --flags`.** `run.py` uses Hydra's compose API,
  not `@hydra.main`, whose command line fails on Python 3.14 ("badly
  formed help string"). `train.py --flags` still works and runs the same
  thing. It translates the flags into overrides, composes the same tree,
  and refuses to run unless the result equals what its own parser gives.
  - A flag that does not apply to the run is now an error rather than
    silently ignored, for example `--turn-n` on a z2 task or `--cooldown`
    without `--schedule wsd`.
  - The cluster job files keep working. The job generators (`scaling.py`,
    `tools/branch_jobs.py`) now write `run.py` commands, and their
    `--extra` takes Hydra overrides (`task.query_times=uniform`), not
    train.py flags.
- **Dependencies.** `requirements.txt` pins `hydra-core==1.3.5` and
  `omegaconf==2.3.1`; the cluster venv needs them (`pip install -r
  requirements.txt` once, after `scripts/sync.sh push`). Every job file
  `scaling.py` and `tools/branch_jobs.py` write checks the import first
  and stops with that instruction.

## The tree

| group / node | options | object (`mapping.config`) | fields |
| --- | --- | --- | --- |
| `model` (required) | `<task>/<name>`, one file per registered model | `Model` | `name`, `ladder`, `size` (a rung of the ladder), `size_kwargs` (resolved from `conf/ladders/<ladder>.yaml`), `arch` (the class, `_partial_`, with fixed arguments such as hybrid45b's `placement`), `query_mode` (`mixed` \| `inert`, below) |
| `task` (chosen by the model; `task=<variant>` overrides) | `z2`, `se2d`, `se2c` (each includes `_episode`) | `Task` | `name`, `episode_steps`, `queries`, `query_times` |
| `generator` (chosen by the task; `generator=<variant>` overrides) | `bank` (z2, se2d), `se2c` | `BankGenerator`, `Se2cGenerator` | bank: `texture bank step_px odometry_dropout workers`; se2c: `texture turn_n motion directions v_min v_max omega_max holdout holdout_mode query_coverage` |
| `schedule` | `cosine`, `wsd` | `Cosine`, `WSD` (`Schedule.lr(step, steps, peak)`) | `warmup` (+ `cooldown` for wsd) |
| `optim` | in `config.yaml` | `AdamW` | `lr` (default: the model's, via `${model_lr:...}`), `weight_decay`, `clip`, `batch` |
| `run` | in `config.yaml` | `Run` | `name out root dir note tag provenance.max_untracked_mb steps seed save_every branch device bf16 compile overwrite` |
| `eval` | in `config.yaml` | `Eval` | `batches every log_every` |
| `experiment` | `se2c_bridge`, `se2d_scaling_wsd`, or any file via `--exp` | none: it only sets the fields above | |

**Each model file selects its own task.** Model names repeat across tasks
(`hybrid4` exists for z2 and for se2d), and a model is written for one
task's interface, so `model=se2d/hybrid4` is the whole choice.

**Size ladders live in `conf/ladders/`,** one file per ladder (`width`,
`vanilla`, `vanilla2`, `lstm`, `mlp`, `depth`), one rung per line in order.
A model file names its ladder (`model.ladder`) and a default rung; pick
another with `model.size=<rung>`. The rung's arguments are resolved into
`model.size_kwargs` and saved with the run. Each model module reads the
same file (`SIZES = mapping.models.ladder('<ladder>')`), so `build(size,
spec)`, the path loaders and tools use, gives the same numbers.
- **A model's architecture is its name at a rung.** An override that
  would train anything else is refused: a ladder argument
  (`+model.arch.width=96`), a rung's numbers (`model.size_kwargs.*`), or
  a fixed argument the module does not register (`ARCH`, e.g. a
  hybrid45b placement). Add a rung to the ladder file, or register a new
  model ("How to add" below).
- `tests/sizes_snapshot.json` holds the pre-move tables; the tests check
  the ladder files against them entry for entry and in order, and that
  config-built models equal `build(size, spec)` models parameter for
  parameter, at every size of every model.

**`model.query_mode`: how query tokens mix** (mapping/models/query_mode.py,
2026-10-17). `mixed`, the default, is every model as before: query tokens
are ordinary tokens (they write the Gated DeltaNet state, sit in the short
convolution's windows, and attention is full). `inert` makes them
read-only: beta = 0 and decay g = 0 for query tokens in every delta-rule
head and both directions; the short convolution of history tokens runs
over the history-only sequence and query tokens get none; attention is
masked so history attends to history only and a query's tokens to history
and their own query (a causal model intersects this with its causal mask);
RoPE is unchanged. History activations then do not depend on the queries,
and a query's answer does not depend on the other queries
(tests/test_query_mode.py).
- Implemented by se2d `hybrid4`, `vanilla4`, `hybrid4_causal`,
  `vanilla4_causal` and se2c `hybrid4c`, `vanilla4c`; each declares it in
  its own class (`QUERY_MODES`). Any other model refuses `inert`, also a
  subclass that inherits the constructor.
- Every masked attention (inert, and causal models in either mode) runs one
  path: FlexAttention with a block mask built by one `mask_mod` from
  per-token metadata (query id, token index), compiled with dynamic shapes,
  so a new query count or episode length does not recompile and memory is
  linear in the tokens (4096 queries on one history: ~120 MiB at xs, as
  unmasked). Full bidirectional attention (mixed, non-causal) needs no mask
  and stays plain SDPA, bit-identical to before.
- `mixed` is not passed to the model, so a mixed run's constructor
  arguments, checkpoint `arch` and args record are byte-for-byte what they
  were before the field (its outputs are bit-identical; tested against the
  commit before it). `inert` adds `query_mode` to `arch` (so checkpoints
  rebuild it) and to the args record. No `train.py` flag (frozen).
- A config.yaml saved before the field has no `model.query_mode`;
  `--from` reads its absence as `mixed`.

## Field map: config key and train.py flag

| config key | train.py flag | | config key | train.py flag |
| --- | --- | --- | --- | --- |
| `model` + `model.size` | `--task --model --size` | | `task.episode_steps` | `--episode-steps` |
| `run.name` | `--run` | | `task.queries` | `--queries` |
| `run.out` | `--out` | | `task.query_times` | `--query-times` |
| `run.steps` | `--steps` | | `task.generator.texture` (bank) | `--texture` |
| `run.seed` | `--seed` | | `task.generator.bank` | `--bank` |
| `run.save_every` | `--save-every` | | `task.generator.step_px` | `--step-px` |
| `run.branch` | `--branch` | | `task.generator.odometry_dropout` | `--odometry-dropout` |
| `run.device` | `--device` | | `task.generator.workers` | `--workers` |
| `run.bf16` | `--no-bf16` (inverted) | | `task.generator.texture` (se2c) | `--se2c-texture` |
| `run.compile` | `--compile` | | `task.generator.turn_n` | `--turn-n` |
| `optim.lr` | `--lr` | | `task.generator.motion` | `--motion` |
| `optim.weight_decay` | `--weight-decay` | | `task.generator.directions` | `--directions` |
| `optim.clip` | `--clip` | | `task.generator.v_min`, `v_max` | `--v-min`, `--v-max` |
| `optim.batch` | `--batch` | | `task.generator.omega_max` | `--omega-max` |
| `schedule` (group) | `--schedule` | | `task.generator.holdout` | `--holdout LO HI` |
| `schedule.warmup` | `--warmup` | | `task.generator.holdout_mode` | `--holdout-mode` |
| `schedule.cooldown` (wsd) | `--cooldown` | | `task.generator.query_coverage` | `--query-coverage` |
| `eval.batches`, `every`, `log_every` | `--eval-batches`, `--eval-every`, `--log-every` | | | |

## Write an experiment

An experiment is a yaml file whose defaults list may pick any group, and
whose values may set only fields that exist. A typo is an error naming the
key. Put it in `conf/experiment/` (`+experiment=<name>`), or anywhere
(`--exp path.yaml`, composed identically):

```yaml
# @package _global_
# One-line purpose, and what it reproduces.
defaults:
  - override /model: se2d/vanilla4
  - override /schedule: wsd

model:
  size: m
run:
  name: seed0_400k_qt_wsd
  steps: 400000
  save_every: 50000
task:
  query_times: uniform
  generator:
    workers: 14
```

Command-line overrides apply on top of it: `--exp f.yaml model.size=l
run.seed=1`. The file is composed under a name of its own, so a file
named like a preset (`se2c_bridge.yaml`) is still your file.

## Variants by composition

A variant of a task or a generator is a small file that includes its
base and changes some values; nothing in Python changes.

```yaml
# conf/generator/se2c_lattice.yaml: the discrete walk on the continuous renderer
# @package task.generator
defaults:
  - se2c
  - _self_
directions: lattice
turn_n: 4
v_min: 9.0
v_max: 9.0
```

```yaml
# conf/task/se2c_long.yaml: se2c with 64-step episodes
# @package _global_
defaults:
  - se2c
  - _self_
task:
  episode_steps: 64
```

Select one on the command line (`generator=se2c_lattice`, `task=se2c_long`)
or in an experiment (`- override /generator: se2c_lattice`), whatever
the model file chose. A variant for the wrong task is an error (a se2c
generator on a z2 task, a se2d task under a se2c model).

## How to add

Each recipe lists every file to touch. After any of them, run
`python run.py <your choice> --print` and `python -m pytest
tests/test_config.py tests/test_models.py`; the config tests check that
every field you added is read by the trainer and changes what it trains
on.

**A model.**
1. `mapping/models/<registry>/<name>.py`, where the registry is
   `mapping/models/` (z2), `mapping/models/se2d/` or
   `mapping/models/se2c/`. Copy a sibling (README.md, "Adding an
   architecture", and CONTRACT.md say what a model owes). It defines:
   - the model class, `__init__(self, spec, <size arguments>, <fixed
     arguments>)`;
   - `SIZES = ladder('<ladder>')`;
   - `SCALING_RULE`;
   - optionally `LR = {rung: lr}` and `NO_DECAY(name) -> bool`;
   - `ARCH = {...}`, the fixed constructor arguments beyond the size (only
     if it has any; e.g. hybrid45b's placement);
   - `build(size, spec)`, returning the class at `SIZES[size]` and `ARCH`
     (the loaders' fallback for checkpoints without `arch`);
   - `register('<name>')(sys.modules[__name__])` at the bottom.
2. `conf/model/<task>/<name>.yaml`. Copy a sibling: `/task`, then
   `model.name` (the file name and the registered name), `model.ladder`,
   a default `model.size`, `model.arch._target_` (the class) with
   `_partial_: true`, and the same fixed arguments as `ARCH`.
3. Its sizes: an existing ladder, or a new `conf/ladders/<ladder>.yaml`
   (one rung per line, smallest first: `xs: {width: 64, depth: 2}`).

A variant of an existing model ("hybrid4c with one change") is a new
model: a new module (a subclass, importing the parent's pieces, as
`vanilla4c.py` does), its own name and file. Overriding the parent's
`arch` from the command line is refused, because the run would be
recorded as the parent.

**A rung.** One line in its ladder file. Every model on that ladder
gets it.

**A se2c generator setting.**
1. A field in `mapping.se2c.generator.Config`, with a default that is
   the generator's behaviour before the setting, used where the walk or
   the queries need it.
2. The same field in `Se2cGenerator` (`mapping/config.py`).
3. Its value in `conf/generator/se2c.yaml`.

`Se2cGenerator.config` passes it to `Config` by name. It reaches the args
record and the tools (`task_se2c.config_from_args`) with no code; records
from before it get the `Config` default. A `train.py` flag is optional
(`task_se2c.add_args` and `train.FLAG_KEYS`); without one, `train.py
--flags` runs take the config default and `--print` says no `train.py`
command is exact.

**A tile-bank generator setting (z2, se2d).** A field in `BankGenerator`
and `conf/generator/bank.yaml`. Then pass it on: `BankGenerator.data` →
`tasks.config_and_spec` (a new argument) → a field of `mapping.data.Config`
and its use. For the tools, `tasks.config_from_args` reads it from the
args record, with the old behaviour as the default.

**A task (episode) setting.** A field in `Task`, its default in
`conf/task/_episode.yaml` (every task includes it), and its use: pass it
to both generators (`BankGenerator.data` and `Se2cGenerator.config`), or
to the trainer (`train.main` reads `e.task`).

**A schedule.**
1. A dataclass in `mapping/config.py` subclassing `Schedule`, with its
   fields, `name = '<name>'` and `lr(self, step, steps, peak)`.
2. `conf/schedule/<name>.yaml` with `_target_` and the fields.

`schedule=<name>` selects it; the tests pick up the file. `train.py
--schedule` offers only cosine and wsd.

**An optimiser setting.** A field in `AdamW` and `conf/config.yaml`
(`optim:`), and its use where `train.main` builds the optimizer. If it
must not change in a branch, leave it out of `train.BRANCH_MAY_CHANGE`.

**An experiment.** A file (above); a preset if it should live in
`conf/experiment/`.

**Known limit.** A setting that matters only under another setting's
value (a persistent-walk option that is moot with `directions=lattice`)
is still read, so the tests cannot tell a run that ignores it. Say so in
the field's comment.

## Sweeps

`run.py` launches one run; a sweep is one command per point.
- **Distinct outputs.** A run directory is `results/<task>/<model>/<size>/<run.name>`
  (`run.dir`, below), and a non-empty one is refused. Points that
  differ only in a value need distinct names. An interpolated name does
  it: `'run.name=lr${optim.lr}_s${run.seed}'` (quoted for the shell).
- **Cluster arrays.** `python scaling.py --task <t> --model <m> --slurm
  --sizes xs s m --lr-grid 1e-3 3e-4 --extra "<overrides>"` writes a SLURM
  array of `run.py` commands, one per (size, lr), named
  `seed<seed>[_lr<lr>]<tag>`. `tools/branch_jobs.py` writes cooldown
  branches. Both check the venv's dependencies first.
- **By hand.** A job script calls `python run.py --exp <file> model.size=$SIZE
  "run.name=<name>_$SIZE"` from the repository root.

## CPU and GPU

Runs read `data/` and write `results/` relative to the working directory:
run from the repository root (in a git worktree, link `data/` in). The
hybrid4 family and hybrid4c run flash-linear-attention's Triton kernels,
which need CUDA; on CPU they are refused before anything is written
(`mapping.models.cuda_only` names the module). The other models train on
CPU (`run.device=cpu`), and the bit-identical rerun guarantee is for CPU.
`run.bf16` applies on CUDA only.

## Run directories

Where a run writes is two config fields, the same for every task:

```yaml
run:
  root: results
  dir: ${run.root}/${task.name}/${model.name}/${model.size}/${run_name:${run.name},${run.seed}}
  out: null            # the explicit override; wins over dir when set
```

- `run_name` is `run.name`, or `seed<seed>` when the name is null (the
  trainer's default name, as before).
- `run.out` stays the explicit override (`train.py --out`); the
  equivalence tests and `--reproduce` write through it.
- `run.root=/elsewhere` moves every run under another root; a preset may
  restructure `run.dir` (`${run.root}/campaigns/h1c/${model.name}/...`).
- `run.root` and `run.dir` say where a run is written, never what it
  trains, so they are not in the legacy `args` record (as `run.overwrite`).
- `mapping.config.Run.location()` gives `{dir, root, rel}`, the resolved
  directory and its path under the root. Tools that need a task's runs use
  `mapping.tasks.results_dir(task)` (`results/<task>`), never a per-task
  root: `mapping/tasks.py` has none any more.

**z2 moved (2026-10-16).** Before the template, z2 runs were written to
`results/<model>/...` with no task segment, the only task without one. All
local z2 runs were moved to `results/z2/<model>/...` (a plain move;
`results/` is untracked). The manifest of every moved entry and run is
`experiments/audit/z2_results_migration.tsv` (a copy is
`results/z2/MIGRATION.tsv`), made by `experiments/audit/z2_results_migration.py`.
Nothing reads the old z2 layout any more. Two read-only allowances remain,
for old records only:
- `scripts/sync.sh pull` maps a remote tree that still has the old layout
  (Torch's `/scratch/ax2348/mapping`) into `results/z2/`, so a pull never
  re-creates it here;
- `run.py --from` on a config.yaml saved before `run.root`/`run.dir`
  existed adds them with today's defaults, without counting them as drift.

Paths recorded inside old `final.json`/`log.jsonl` files (`args.out`, null
for default directories) are unchanged; the files were moved, not edited.

## Provenance: every run's code and record

Every run is pinned before it trains (mapping/provenance.py;
docs/provenance_design.md, §15 has what was built):
- **A snapshot commit** of the exact working tree (tracked and untracked
  files, `.gitignore` respected, `experiments/ledger/` left out) at
  `refs/runs/<run_id>`, made through a private index: the branch, HEAD and
  the index are never touched. A clean tree's snapshot is HEAD. A dirty
  tree is fine (no strict mode): the snapshot holds its bytes. An untracked
  file over `run.provenance.max_untracked_mb` (20) is refused, and so is a
  run outside a git checkout or with no commit.
- **A ledger record**, `experiments/ledger/<host>/<run_id>.json` in the main
  worktree (`XP_LEDGER_DIR` overrides it), with the run's code, command,
  resolved config, note, environment, data hashes, Slurm ids, progress and,
  at the end, its status and final metrics. The run directory holds the
  same record as `provenance.json`, its id in `RUN_ID` (claimed when the
  directory is created, so two runs can never write one directory) and
  `pip_freeze.txt`. Every checkpoint carries `provenance: {run_id, snapshot}`.
- `run_id` is `<UTC yyyymmddTHHMMSSZ>-<host>-<6 hex>`; the host is
  `XP_HOST`, else the machine name mapped by `conf/hosts.yaml`.
- See a run's code: `git show refs/runs/<run_id>`; check it out with
  `git worktree add --detach <dir> refs/runs/<run_id>`.

## xp: the porcelain

`xp` (mapping/xp/, `pip install -e .` once, or `python -m mapping.xp`)
reads the ledger and wraps git and run.py; it never trains itself.
`--explain` prints every underlying command; `--dry-run` prints those that
would change something and runs none; queries take `--json`.

```
xp ls                         the latest 20 runs, newest first: start, status, note, path,
                              host, tag, final mse (retroactive records marked *)
xp ls --all --running --tag h1c --task se2d --model hybrid4 --host torch --since 2d --json
xp show <run>                 one run: record, snapshot (git show --stat), dirty diff, files,
                              registry entry, reproductions
xp compare <run> <run> ...    final metrics, config differences, code differences (--grid)
xp run <preset> -m '<note>' [overrides] [--tag T] [--bg]     python run.py +experiment=<preset>
xp reproduce <run>            the run's exact code in a temporary worktree, into
                              <run>/reproductions/<UTC ts>_<new id>/, with reproduce_report.json
xp sync                       commit experiments/ledger/ (by explicit path; local only for now)
xp tag <run> <tag> --note ..  append to experiments/registry.yaml (tagged:)
```

A `<run>` is its id, a unique prefix or 6-hex suffix of it, or its results
path (`z2/vanilla4/xs/a`, `torch:se2d/hybrid4/s/seed0_400k_qt_wsd`).

**Reproductions.** `xp reproduce` checks out `refs/runs/<id>` as a detached
worktree, links the data files the run read (refusing if their sha256
differs from the ledger's, unless `--allow-data-drift`), and runs that
worktree's own `run.py --from <the run's config>` with
`run.out=<run>/reproductions/<UTC ts>_<new id>/` and
`run.note='reproduce <id>'`. The new run's ledger record says
`"reproduces": "<id>"`, and its snapshot is the original's (the worktree is
clean at it). `reproduce_report.json` compares final metrics, checkpoint
tensors (tools/compare_checkpoints.py), code, environment, data and config;
its verdict is `bitwise` (identical checkpoint tensors and metrics),
`within_tolerance` (`--tolerance`, 1e-6) or `differs`. A run not on this
machine gets a stand-in directory marked `STUB`, holding its record.
summarize.py, scaling.py's fits and `mapping.provenance.iter_run_dirs` never
count anything under `reproductions/`. A retroactive record has no snapshot
and is refused.

## Outputs and reruns

- **What a run directory holds besides the results.** `config.yaml`, the
  resolved config (interpolations evaluated, so `optim.lr` is a number
  and `model.size_kwargs` the rung's arguments); `overrides.yaml`, what
  it was composed from; and `experiment.yaml`, a copy of its experiment
  file when it had one. Nothing is written until the data, the model and
  the device have been checked, so a run that fails at startup leaves no
  directory.
- **Never overwritten.** A run directory that is not empty is refused.
  Give the run another `run.name`, or replace it with
  `run.overwrite=true` (`train.py --overwrite`). Files of the old run
  that the new one does not write (a `step<k>.pt` past the new run's
  length) stay. `run.overwrite` is not saved in `overrides.yaml`, so a
  rerun never inherits it.
- **Rerunning.** `python run.py --from <run>/config.yaml [overrides ...]`
  recomposes the run from its `overrides.yaml` and `experiment.yaml` plus
  the new overrides, so group choices (`schedule=wsd`) and sizes
  (`model.size=s`) apply and what follows from them (the rung's
  arguments, the model's default learning rate) follows. A rerun's log is
  bit-identical to the original's on CPU.
  - Recomposing the sources unchanged must give the saved config. If
    `conf/` changed since the run (a default, a ladder, a model's `LR`),
    the rerun is refused and the changed keys are listed: rerun from the
    commit the run was made at, or write a new experiment.
  - Runs saved before `overrides.yaml` existed take value overrides of
    existing keys only, merged into their saved config.
  - A rerun needs a new `run.name` (or `run.out`), since the original's
    directory is not empty, and its own `run.note`.
- **The legacy `args`.** Checkpoints, `final.json` and the first record
  of `log.jsonl` keep the flat `args` dict as before, with the same keys
  and values as a `train.py --flags` run, so every tool and loader reads
  new runs unchanged. It is an output only: the trainer never reads it.
  Every task, generator and schedule field is copied into it under its
  own name (the se2c texture as `se2c_texture`), so a new field reaches
  it with no code. Settings a run's variants do not compose appear there
  at train.py's defaults, for example the se2c generator fields on a z2
  run.
- **`arch` in checkpoints.** `final.pt` and `step<k>.pt` also hold
  `arch`, the model's class and constructor arguments
  (`Model.record()`). `load_checkpoint` builds from it, and falls back to
  the registered module's `build(size, spec)` for older checkpoints.

## Branching

`run.branch=<parent>/step<k>.pt` continues a run from a saved step
(`train.py --branch`), typically with a new length and schedule. A
branch must continue the same run: when the parent's `config.yaml` is
beside the checkpoint, every key except `run.*`, `eval.*`, `schedule.*`,
`optim.weight_decay`, `optim.clip` and `task.generator.workers` must be
the parent's, or the branch is refused with the keys listed. Older
parents without `config.yaml` get the old check of their `args`
(task, model, size, batch, seed, lr, and the bank generator's settings).

## se2 is now se2d

The discrete SE(2) task was renamed `se2d` (2026-10-16).
- New runs go to `results/se2d/`.
- **Old se2 runs stay in `results/se2/`, not moved (decided with the run
  directory template, 2026-10-16).** Their path already is the template,
  `${run.root}/${task.name}/...`, applied to the task name they were
  recorded with (`se2` in their `final.json`, checkpoints and log). The
  registry, the audit and published reports cite these paths, and the
  Torch trees hold the same layout. So a run's directory is always the
  template over its *recorded* task name, and only z2, whose old path had
  no task segment at all, had to move. `mapping.tasks.results_dir('se2')`
  is `results/se2`.
- `se2` remains a read-only alias (`mapping.tasks.ALIASES`). Old
  checkpoints and `final.json` files load, and `load_checkpoint` gives
  their task as `se2d`, so the se2d tools (probe_gdn, read_beta, ...)
  read them. `summarize.py --task se2` reads `results/se2/`, and a new se2d
  run may branch from an old se2 checkpoint. A new run cannot be launched
  as `se2`.
