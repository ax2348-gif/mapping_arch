# Configuration (Hydra)

A run is configured by composing the tree in `conf/` and instantiating it
into the objects in `mapping/config.py`. The rule: **every field in a
composed config is consumed by an object**.
- Variants (which generator a task uses, which schedule, which size) are
  chosen by the defaults list, so a run never carries the fields of a
  variant it does not use.
- Every node instantiates a class with a strict signature, so an unknown
  or misspelled key is an error, not an ignored setting.

`tests/test_config.py` enforces both. It changes every field of
representative configs in turn and checks that the run changes.

Behaviour is unchanged from the `train.py --flags` harness:
`tests/test_equivalence_runs.py` trains short runs on the pre-refactor
code and on both entry points here, and compares everything they write.

## Launch

```
python run.py model=se2d/hybrid4                      # task se2d, size xs, cosine schedule
python run.py model=se2d/hybrid4 model/size=m schedule=wsd run.steps=400000
python run.py model=z2/vanilla model/size=vanilla/l   # a model on its own ladder
python run.py +experiment=se2c_bridge model/size=s    # a preset (conf/experiment/)
python run.py --exp ~/notes/my_run.yaml run.seed=1    # your own experiment file
python run.py --from results/se2c/hybrid4c/s/bridge_n4dec_100k_qt/config.yaml run.name=again
python run.py ... --print                             # resolved config + train.py flags; no run
```

- **Arguments.** Hydra overrides: `key=value`, `group=option` and
  `+experiment=name`.
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
| `model` (required) | `<task>/<name>`, one file per registered model | `Model` (name, size, `arch`), with `arch` the model class (`_partial_`) | `name`; `arch.*` = the class's own constructor arguments |
| `model/size` (chosen by the model file) | `xs` `s` `m` `l` `xl` `xxl` (width ladder); `vanilla/*`, `vanilla2/*`, `lstm/*`, `mlp/*`, `depth/d4..d32` | merged into `model` | `size`, `arch.width` (+ `arch.depth`) |
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

**Size ladders live in `conf/model/size/`.** Each model module's `SIZES`
is read from there (`mapping.models.ladder`), so `build(size, spec)`, the
path used by loaders and tools, uses the same numbers. Each module's
`SCALING_RULE` still describes its ladder. `tests/sizes_snapshot.json`
holds the pre-move tables, and the tests check the files against them
entry for entry and in order. They also check that config-built models
equal `build(size, spec)` models parameter for parameter, at every size of
every model.

## Field map: config key and train.py flag

| config key | train.py flag | | config key | train.py flag |
| --- | --- | --- | --- | --- |
| `model` + `model/size` | `--task --model --size` | | `task.episode_steps` | `--episode-steps` |
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
  - override /model/size: m
  - override /schedule: wsd

run:
  name: seed0_400k_qt_wsd
  steps: 400000
  save_every: 50000
task:
  query_times: uniform
  generator:
    workers: 14
```

Command-line overrides apply on top of it: `--exp f.yaml model/size=l
run.seed=1`.

## Outputs and reruns

- **`config.yaml`.** Each run directory gets this file, the resolved
  config (interpolations evaluated, so `optim.lr` is a number).
- **Rerunning.** `python run.py --from <run>/config.yaml [key=value ...]`
  reruns it. Only value overrides apply there, since group choices are
  already resolved. A rerun's log is bit-identical to the original's on
  CPU.
  - Without an override of `run.name` or `run.out`, it writes to the same
    directory as the original.
- **The legacy `args`.** Checkpoints, `final.json` and the first record
  of `log.jsonl` keep the flat `args` dict as before, with the same keys
  and values as a `train.py --flags` run, so every tool and loader reads
  new runs unchanged. It is an output only. Settings a run's variants do
  not compose appear there at train.py's defaults, for example the se2c
  generator fields on a z2 run.

## se2 is now se2d

The discrete SE(2) task was renamed `se2d` (2026-10-16).
- New runs go to `results/se2d/`.
- `se2` remains a read-only alias (`mapping.tasks.ALIASES`). Old
  checkpoints and `final.json` files load, `summarize.py --task se2` reads
  `results/se2/`, and a new se2d run may branch from an old se2
  checkpoint. A new run cannot be launched as `se2`.
