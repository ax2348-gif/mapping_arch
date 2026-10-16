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
python run.py model=se2d/hybrid4                      # task se2d, size xs, cosine schedule
python run.py model=se2d/hybrid4 model.size=m schedule=wsd run.steps=400000
python run.py model=z2/vanilla model.size=l            # a rung of the model's own ladder
python run.py +experiment=se2c_bridge model.size=s    # a preset (conf/experiment/)
python run.py --exp ~/notes/my_run.yaml run.seed=1    # your own experiment file
python run.py --from results/se2c/hybrid4c/s/bridge_n4dec_100k_qt/config.yaml run.name=again schedule=wsd
python run.py ... --print                             # resolved config, and the exact train.py command if any; no run
```

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
  - The cluster job files keep working.
- **Dependencies.** The cluster venv needs `hydra-core` and `omegaconf`
  for this tree.

## The tree

| group / node | options | object (`mapping.config`) | fields |
| --- | --- | --- | --- |
| `model` (required) | `<task>/<name>`, one file per registered model | `Model` | `name`, `ladder`, `size` (a rung of the ladder), `size_kwargs` (resolved from `conf/ladders/<ladder>.yaml`), `arch` (the class, `_partial_`, with fixed arguments such as hybrid45b's `placement`) |
| `task` (chosen by the model) | `z2`, `se2d`, `se2c` | `Task` | `name`, `episode_steps`, `queries`, `query_times` |
| `generator` (chosen by the task) | `bank` (z2, se2d), `se2c` | `BankGenerator`, `Se2cGenerator` | bank: `texture bank step_px odometry_dropout workers`; se2c: `texture turn_n motion directions v_min v_max omega_max holdout holdout_mode query_coverage` |
| `schedule` | `cosine`, `wsd` | `Cosine`, `WSD` | `warmup` (+ `cooldown` for wsd) |
| `optim` | in `config.yaml` | `AdamW` | `lr` (default: the model's, via `${model_lr:...}`), `weight_decay`, `clip`, `batch` |
| `run` | in `config.yaml` | `Run` | `name out steps seed save_every branch device bf16 compile` |
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
    directory is not empty.
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
- `se2` remains a read-only alias (`mapping.tasks.ALIASES`). Old
  checkpoints and `final.json` files load, and `load_checkpoint` gives
  their task as `se2d`, so the se2d tools (probe_gdn, read_beta, ...)
  read them. `summarize.py --task se2` reads `results/se2/`, and a new se2d
  run may branch from an old se2 checkpoint. A new run cannot be launched
  as `se2`.
