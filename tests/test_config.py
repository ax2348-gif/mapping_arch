"""The Hydra config tree (conf/, mapping/config.py, run.py).

1. Every field is consumed. Every composable config (each model file,
   each schedule, each experiment preset) instantiates; an unknown key at
   any node is an error; a short real run of the trainer reads every
   field of every object; and changing any value changes what the trainer
   trains on (through the calls it makes: the generator config, the
   learning rate, the model it builds), or is refused. So no field is
   ever composed that nothing reads.
2. The size ladders moved into conf/ladders/ are the old in-code
   tables, entry for entry and in order (tests/sizes_snapshot.json), and
   a model file's ladder is its module's. An override that would train
   another architecture than the registered model's is refused.
3. Models built from the config are the old ``build(size, spec)`` models,
   parameter for parameter, for every model and size.
4. ``train.py --flags`` and ``run.py`` give the same settings.
Behaviour of whole runs against the pre-refactor code is in
tests/test_equivalence_runs.py.
"""
import copy
import glob
import json
import os

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

import mapping.tasks as tasks
import run
import train
from mapping.config import model_lr

MODELS = sorted(os.path.relpath(f, 'conf/model')[:-5]
                for t in ('z2', 'se2d', 'se2c') for f in glob.glob(f'conf/model/{t}/*.yaml'))
PRESETS = sorted(os.path.basename(f)[:-5] for f in glob.glob('conf/experiment/*.yaml'))
SNAPSHOT = json.load(open('tests/sizes_snapshot.json'))


def compose(*overrides):
    return run.resolved(run.load(list(overrides)))


def leaves(node, path=()):
    """(path, value) of every leaf a run consumes (not Hydra's own keys)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in ('_target_', '_partial_'):
                yield from leaves(v, path + (k,))
    else:
        yield path, node


def test_every_model_has_a_config_and_back():
    registered = sorted(f'{t}/{n}' for t in ('z2', 'se2d', 'se2c')
                        for n in tasks.get(t)['models'].REGISTRY)
    assert MODELS == registered


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('schedule', ['cosine', 'wsd'])
def test_every_config_instantiates(model, schedule):
    cfg = compose(f'model={model}', f'schedule={schedule}')
    exp = run.build(cfg)
    a = exp.legacy_args()
    assert (a.task, a.model, a.schedule) == (*model.split('/'), schedule)
    # the generator composed is the one the task uses, and only that one
    gpu = tasks.get(a.task).get('gpu_data', False)
    assert ('turn_n' in cfg.task.generator) == gpu
    assert ('step_px' in cfg.task.generator) == (not gpu)
    assert ('cooldown' in cfg.schedule) == (schedule == 'wsd')


@pytest.mark.parametrize('preset', PRESETS)
def test_presets_instantiate(preset):
    run.build(compose(f'+experiment={preset}'))


def _bump(v):
    """A different value of the same kind."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v + 0.25
    if isinstance(v, str):
        return v + 'x'
    if v is None:
        return 'x'
    if isinstance(v, list):
        return [x + 0.25 for x in v] if v else ['x']
    raise TypeError(v)


CONSUMED_CASES = (
    [[f'model={m}'] for m in MODELS]
    + [['model=z2/hybrid4', f'schedule={os.path.basename(f)[:-5]}']
       for f in sorted(glob.glob('conf/schedule/*.yaml'))]
    + [[f'+experiment={p}'] for p in PRESETS])


# fields that are not settings of the run, so have no place in the args
# record (it must stay what train.py --flags gives): config.yaml has them.
# run.root and run.dir say where the run is written (the args record keeps
# train.py's ``out``, null for a default directory, as before them).
# run.note, run.tag and run.provenance are launch metadata, recorded in the
# provenance ledger only (tests/test_provenance.py: they change no training input).
NOT_IN_RECORD = {'run.overwrite', 'run.root', 'run.dir', 'run.note', 'run.tag',
                 'run.provenance.max_untracked_mb'}


def trainer_inputs(e):
    """What ``train.main`` derives from the objects, through the calls it
    makes: the generator config (and spec, bank file and loader workers),
    the learning rate at every step, the model it builds, and the
    optimizer, run and evaluation settings."""
    from dataclasses import asdict
    from mapping.config import BankGenerator
    g, t = e.task.generator, e.task
    if isinstance(g, BankGenerator):
        config, spec = g.data(t, e.run.seed)
        data = (config, spec.__dict__, g.bank_path(), g.workers)
    else:
        data = g.config(t, e.run.seed)
    steps = sorted({*range(0, e.run.steps, max(1, e.run.steps // 400)), e.run.steps - 1})
    lrs = [e.schedule.lr(i, e.run.steps, e.optim.lr) for i in steps]
    return (t.name, data, lrs, e.model.record(), asdict(e.optim), asdict(e.run),
            asdict(e.eval))


@pytest.mark.parametrize('overrides', CONSUMED_CASES, ids=lambda o: ' '.join(o))
def test_every_field_is_consumed(overrides):
    """Change each leaf in turn: what the trainer trains on must change,
    or the change must be refused (an architecture argument, a name). The
    args record must change too, so the tools that read it see it."""
    base = OmegaConf.to_container(compose(*overrides))
    exp = run.build(OmegaConf.create(base))
    t0, a0 = trainer_inputs(exp), vars(exp.legacy_args())
    for path, value in leaves(base):
        cfg = copy.deepcopy(base)
        node = cfg
        for k in path[:-1]:
            node = node[k]
        if path[0] == 'model' and path[1] in ('arch', 'size_kwargs'):
            node[path[-1]] = (value * 2 if isinstance(value, int) else
                              (['first'] if value != ['first'] else ['last']))
        else:
            node[path[-1]] = _bump(value)
        try:
            e = run.build(OmegaConf.create(cfg))
            t = trainer_inputs(e)
        except Exception:
            continue          # refused outright, before any training: loud
        name = '.'.join(path)
        assert t != t0, f'{name} changed but what the trainer trains on did not'
        if name not in NOT_IN_RECORD:
            assert vars(e.legacy_args()) != a0, f'{name} changed but the args record did not'


TRACKED_RUNS = {
    'z2 cosine': ['model=z2/vanilla4', 'task.generator.workers=0'],
    'se2d wsd': ['model=se2d/vanilla4', 'task.generator.workers=0', 'schedule=wsd',
                 'task.generator.odometry_dropout=0.5'],
    'se2c': ['model=se2c/vanilla4c', 'task.generator.holdout=[0.8,1.2]',
             'task.generator.turn_n=8', 'task.generator.odometry_dropout=0.5'],
}


@pytest.mark.parametrize('case', sorted(TRACKED_RUNS))
def test_the_trainer_reads_every_field(tmp_path, monkeypatch, case):
    """A short real run of ``train.main`` with every config object's
    attribute reads recorded (not those that only build the args record,
    or that dataclass machinery makes): each field of each object must be
    read by the trainer."""
    import sys
    from dataclasses import fields
    from mapping.config import NODES
    skip = {'legacy_args', '__post_init__', '__repr__', '__eq__', '__hash__', '__init__'}
    cfg = compose(*TRACKED_RUNS[case], 'run.note=test', 'run.device=cpu', 'run.steps=2',
                  'run.save_every=1',
                  'eval.every=1', 'eval.log_every=1', 'eval.batches=1', 'optim.batch=2',
                  'schedule.warmup=1', f'run.root={tmp_path}')   # run.out unset: the template is read
    exp = run.build(cfg)
    reads = set()
    for cls in NODES:
        names = {f.name for f in fields(cls)}

        def tracked(self, key, _names=names):
            if key in _names and sys._getframe(1).f_code.co_name not in skip:
                reads.add((type(self).__name__, key))
            return object.__getattribute__(self, key)
        monkeypatch.setattr(cls, '__getattribute__', tracked)
    train.main(exp, config=cfg)
    nodes = (exp.run, exp.eval, exp.optim, exp.schedule, exp.task, exp.task.generator)
    want = {(type(n).__name__, f.name) for n in nodes for f in fields(n)}
    assert want - reads == set(), f'never read by the trainer: {sorted(want - reads)}'


@pytest.mark.parametrize('node', ['', 'optim', 'run', 'eval', 'schedule', 'task',
                                  'task.generator', 'model', 'model.arch'])
def test_unknown_keys_are_errors(node):
    cfg = compose('model=se2d/hybrid4')
    target = OmegaConf.select(cfg, node) if node else cfg
    target['not_a_field'] = 1        # composed configs are saved unlocked
    with pytest.raises(Exception, match='not_a_field'):
        run.build(cfg)


def test_ladders_are_the_old_tables():
    """The ladders moved out of the modules keep the old tables: every old
    rung, with its arguments, in the old order (rungs added since may sit
    anywhere). Models registered since (not in the snapshot) have no old
    table; every snapshot model must still be registered."""
    registered = {f'{t}/{n}': m for t in ('z2', 'se2d', 'se2c')
                  for n, m in tasks.get(t)['models'].REGISTRY.items()}
    assert set(SNAPSHOT) <= set(registered)
    for key, rows in SNAPSHOT.items():
        sizes = registered[key].SIZES
        assert all(sizes.get(k) == v for k, v in rows), key
        old = [k for k in sizes if k in dict(rows)]
        assert old == [k for k, _ in rows], key


def test_model_files_name_their_ladder():
    """conf/model/<task>/<name>.yaml at each rung gives the module's SIZES
    entry (plus its fixed ARCH), and the rung's name."""
    for model in MODELS:
        task, name = model.split('/')
        module = tasks.get(task)['models'].get(name)
        for size, kwargs in module.SIZES.items():
            m = run.build(compose(f'model={model}', f'model.size={size}')).model
            assert m.size == size and m.size_kwargs == kwargs, (model, size)
            fixed = {k: tuple(v) if isinstance(v, (list, tuple)) else v
                     for k, v in getattr(module, 'ARCH', {}).items()}
            assert m.kwargs == {**kwargs, **fixed}, (model, size)


@pytest.mark.parametrize('model', MODELS)
def test_config_models_are_the_old_models(model):
    """Same seed, same parameters, bit for bit, at every size."""
    task, name = model.split('/')
    module = tasks.get(task)['models'].get(name)
    spec = tasks.config_and_spec(task, 32, 8, 0, 'uniform')[1]
    overrides = ['model.query_mode=inert'] if name in ('hybrid7', 'hybrid7c', 'hybrid7_odom') else []
    for size in module.SIZES:
        if overrides and size == 'xs':
            with pytest.raises(ValueError, match='size S or larger'):
                module.build(size, spec)
            continue
        torch.manual_seed(0)
        old = module.build(size, spec).state_dict()
        torch.manual_seed(0)
        new = run.build(compose(f'model={model}', f'model.size={size}', *overrides)).model.build(spec)
        new = new.state_dict()
        assert old.keys() == new.keys(), (model, size)
        assert all(torch.equal(old[k], new[k]) for k in old), (model, size)
        assert type(module.build(size, spec)) is type(run.build(
            compose(f'model={model}', f'model.size={size}', *overrides)).model.build(spec))


@pytest.mark.parametrize('overrides,match', [
    # each of these used to compose, save the override in config.yaml and
    # train the registered model regardless (seams probe, 2026-10-16)
    (['model=se2c/hybrid4c', '+model.arch.width=96'], 'set by the size'),
    (['model=z2/vanilla', '+model.arch.width=96'], 'set by the size'),
    (['model=se2d/hybrid45b_two', 'model.arch.placement=[last]'], 'is not what hybrid45b_two registers'),
    (['model=se2c/hybrid4c', '+model.arch.patch=5'], 'is not what hybrid4c registers'),
    (['model=se2d/hybrid4', '+model.arch.patch=5'], "unexpected keyword argument 'patch'"),
    (['model=z2/vanilla', 'model.size=zz'], 'not a rung of the .vanilla. ladder'),
])
def test_architecture_overrides_are_refused(overrides, match):
    with pytest.raises(Exception, match=match):
        run.build(compose(*overrides))


def test_size_kwargs_must_be_the_rung():
    cfg = compose('model=z2/vanilla', 'model.size=m')
    cfg.model.size_kwargs.width = 192            # the shared ladder's m
    with pytest.raises(Exception, match="not rung 'm' of the 'vanilla' ladder"):
        run.build(cfg)
    cfg = compose('model=z2/vanilla', 'model.size=m')
    cfg.model.size = 's'                         # a relabel of m's numbers
    with pytest.raises(Exception, match="not rung 's'"):
        run.build(cfg)


def test_model_file_and_module_agree_on_the_ladder():
    cfg = compose('model=z2/vanilla', 'model.size=m')
    cfg.model.ladder, cfg.model.size_kwargs = 'width', {'width': 192}
    with pytest.raises(Exception, match="names ladder 'width'"):
        run.build(cfg)


def test_old_size_syntax_explains_itself():
    with pytest.raises(SystemExit, match='model.size=m'):
        run.parse_cli(['model=z2/vanilla', 'model/size=vanilla/m'])


FLAG_CASES = [
    # the cluster job files (scaling study, written when the task was se2)
    '--task se2d --model hybrid4 --size l --run seed0_400k_qt_wsd --steps 400000 --batch 64 '
    '--seed 0 --workers 14 --query-times uniform --schedule wsd --save-every 50000 '
    '--eval-every 10000 --log-every 2000',
    '--task se2d --model vanilla4 --size s --run seed0_400k_qt_wsd --steps 400000 --batch 64 '
    '--seed 0 --workers 14 --query-times uniform --schedule wsd --save-every 50000 '
    '--eval-every 10000 --log-every 2000 --episode-steps 64',
    '--task se2d --model hybrid4 --size s --steps 125000 --batch 64 --seed 0 --workers 14 '
    '--lr 1e-4 --query-times uniform --schedule wsd --eval-every 2500 --log-every 1000',
    # the se2c bridge jobs
    '--task se2c --model hybrid4c --size m --steps 100000 --query-times uniform '
    '--eval-every 2500 --log-every 500 --save-every 5000 --motion decoupled '
    '--directions lattice --turn-n 4 --v-min 9 --v-max 9 --run bridge_n4dec_100k_qt',
    # assorted z2 and options
    '--model vanilla --size m --steps 300',
    '--model vanilla2 --size xl',
    '--model mlp --size l --texture imagenette_hp --step-px 5 --odometry-dropout 1.0',
    '--model vanilla_depth --size d32 --no-bf16 --compile --device cpu',
    '--task se2c --model vanilla4c --size s --holdout 0.8 1.2 --holdout-mode only '
    '--se2c-texture mnist --motion arc --omega-max 1.5 --query-coverage global',
    '--task se2c --model hybrid4c --size m --steps 120000 --query-times uniform '
    '--odometry-dropout 1.0 --run vo_120k_qt',
    '--model hybrid4 --size xs --out /tmp/a\\ b --branch x=y.pt --cooldown 0.2',
]


@pytest.mark.parametrize('line', FLAG_CASES)
def test_flags_and_config_agree(line):
    import shlex
    parsed = train.parse(shlex.split(line))
    args = run.build(run.resolved(run.load(train.overrides_from_flags(parsed)))).legacy_args()
    if parsed.lr is None:
        parsed.lr = model_lr(parsed.task, parsed.model, parsed.size)
    # every setting a flag expresses (a config field with no flag, a new
    # generator setting say, is in args only, at its config default)
    assert {k: vars(args).get(k, '<missing>') for k in vars(parsed)} == vars(parsed)


@pytest.mark.parametrize('line', ['--model hybrid4 --size xs --turn-n 4',
                                  '--task se2c --model hybrid4c --size xs --workers 2',
                                  '--model hybrid4 --size xs --cooldown 0.3',
                                  '--task se2d --model hybrid4c --size xs'])
def test_flags_that_do_not_apply_are_refused(line):
    with pytest.raises(SystemExit):
        train.overrides_from_flags(train.parse(line.split()))


def test_old_task_name_is_not_a_new_run():
    with pytest.raises(SystemExit):
        train.parse('--task se2 --model hybrid4 --size xs'.split())


def test_external_experiment_file(tmp_path):
    exp = tmp_path / 'mine.yaml'
    exp.write_text('# @package _global_\ndefaults:\n  - override /model: se2d/vanilla4\n'
                   '  - override /schedule: wsd\n'
                   'model:\n  size: m\nrun:\n  steps: 123\ntask:\n  generator:\n    texture: imagenette\n')
    a = run.build(run.resolved(run.load(['run.seed=3'], exp=str(exp)))).legacy_args()
    assert (a.task, a.model, a.size, a.schedule, a.steps, a.seed, a.texture) == (
        'se2d', 'vanilla4', 'm', 'wsd', 123, 3, 'imagenette')
    bad = tmp_path / 'typo.yaml'
    bad.write_text('# @package _global_\nrun:\n  stpes: 5\n')
    with pytest.raises(Exception, match='stpes'):
        run.build(run.resolved(run.load(['model=z2/hybrid4'], exp=str(bad))))


TINY = ['run.device=cpu', 'run.steps=2', 'eval.every=2', 'eval.log_every=2',
        'eval.batches=1', 'optim.batch=2', 'run.note=test']


def test_an_experiment_file_named_like_a_preset_is_that_file(tmp_path):
    mine = tmp_path / 'se2c_bridge.yaml'          # conf/experiment/ has one too
    mine.write_text('# @package _global_\ndefaults:\n  - override /model: se2c/vanilla4c\n'
                    'run:\n  name: mine\n')
    cfg = run.resolved(run.load([], exp=str(mine)))
    assert (cfg.model.name, cfg.run.name) == ('vanilla4c', 'mine')


def test_a_run_directory_is_never_overwritten_silently(tmp_path):
    out = f'run.out={tmp_path}/r'
    run.main(['model=se2c/vanilla4c', out, *TINY])
    before = open(tmp_path / 'r' / 'log.jsonl').read()
    with pytest.raises(SystemExit, match='is not empty'):
        run.main(['model=se2c/vanilla4c', out, 'run.seed=1', *TINY])
    assert open(tmp_path / 'r' / 'log.jsonl').read() == before
    with pytest.raises(SystemExit, match='is not empty'):     # --from, same name
        run.main(['--from', str(tmp_path / 'r' / 'config.yaml'), 'run.seed=1', 'run.note=test'])
    run.main(['model=se2c/vanilla4c', out, 'run.seed=1', 'run.overwrite=true', *TINY])
    assert open(tmp_path / 'r' / 'log.jsonl').read() != before


def test_nothing_is_written_before_the_run_is_checked(tmp_path):
    with pytest.raises(SystemExit, match='CUDA-only'):
        run.main(['model=se2c/hybrid4c', f'run.out={tmp_path}/r', *TINY])
    assert not os.path.exists(tmp_path / 'r')


def test_from_recomposes_from_the_saved_sources(tmp_path):
    exp = tmp_path / 'mine.yaml'
    exp.write_text('# @package _global_\ndefaults:\n  - override /model: se2d/hybrid4\n'
                   'run:\n  name: mine\n  steps: 123\n')
    d = tmp_path / 'r'
    cfg = run.resolved(run.load([f'run.out={d}'], exp=str(exp)))
    os.makedirs(d)
    train.write_sources(str(d), cfg, {'overrides': [f'run.out={d}'], 'experiment': str(exp)})
    again, sources = run.rerun(str(d / 'config.yaml'), [f'run.out={tmp_path}/r2'])
    assert OmegaConf.to_container(again) == {**OmegaConf.to_container(cfg),
                                             'run': {**cfg.run, 'out': f'{tmp_path}/r2'}}
    # a group choice and a size: the rung's arguments and the lr follow
    again, _ = run.rerun(str(d / 'config.yaml'), ['schedule=wsd', 'model.size=m'])
    assert again.schedule.cooldown == 0.2 and again.model.size_kwargs == {'width': 192}
    assert again.run.steps == 123 and again.optim.lr == model_lr('se2d', 'hybrid4', 'm')
    # the tree changed since the run: refused, not silently another run
    saved = OmegaConf.load(d / 'config.yaml')
    saved.eval.batches = 99
    OmegaConf.save(saved, d / 'config.yaml')
    with pytest.raises(SystemExit, match='changed: eval.batches'):
        run.rerun(str(d / 'config.yaml'), [])


def test_from_an_older_run_takes_value_overrides_only(tmp_path):
    cfg = compose('+experiment=se2c_bridge', 'model.size=s')
    for k in ('overwrite', 'root', 'dir'):       # saved before these fields existed
        del cfg.run[k]
    path = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, path)
    again, _ = run.rerun(str(path), ['run.seed=7', 'run.name=again'])
    assert run.build(again).legacy_args().seed == 7 and again.run.overwrite is False
    # the run directory template is added, over the saved run's own values
    assert again.run.dir == 'results/se2c/hybrid4c/s/again'
    for bad, match in (['schedule=wsd', 'a group choice cannot be recomposed'],
                       ['model/size=s', 'not group choices'],
                       ['run.sed=1', "no key 'run.sed'"],
                       ['task.generator=1', 'is a config node']):
        with pytest.raises(SystemExit, match=match):
            run.rerun(str(path), [bad])
    again, _ = run.rerun(str(path), ['model.size=xs'])   # relabels m's numbers: refused
    with pytest.raises(Exception, match="not rung 'xs'"):
        run.build(again)


@pytest.mark.parametrize('argv,match', [
    ([], 'choose a model: model=<task>/<name>, one of se2c/hybrid4c'),
    (['model=se2c/hybrid4'], "no model 'se2c/hybrid4'.*have se2c/hybrid4c"),
    (['model=z2/hybrid4', 'run.sed=1'], "run has no field 'sed'; its fields: name, out"),
    (['model=z2/hybrid4', 'model.arch.width=96'], 'model.size=<rung>'),
    (['model=z2/hybrid4', 'run.seed=1,2'], 'run.py launches one run'),
])
def test_composition_errors_say_what_to_do(argv, match):
    with pytest.raises(SystemExit, match=match):
        run.load(argv)


def test_an_experiment_file_needs_its_header(tmp_path):
    f = tmp_path / 'x.yaml'
    f.write_text('run:\n  steps: 5\n')
    with pytest.raises(SystemExit, match='# @package _global_'):
        run.load(['model=z2/hybrid4'], exp=str(f))


def test_print_shows_train_py_only_when_exact(monkeypatch):
    cfg = compose('+experiment=se2c_bridge')
    assert run.flags_equivalent(cfg, run.build(cfg)).startswith('train.py --task se2c')
    # a setting with no flag (as a new generator setting would be)
    monkeypatch.delitem(train.FLAG_KEYS, 'turn_n')
    assert run.flags_equivalent(cfg, run.build(cfg)).startswith(
        '# no exact train.py equivalent (differs: task.generator.turn_n')


def test_flags_translation_ignores_settings_flags_cannot_express(monkeypatch):
    """A config field with no flag (a new generator setting) appears in the
    args record but not in the parsed flags; train.py --flags compares
    only what the flags express, and runs."""
    from mapping.config import Experiment
    legacy = Experiment.legacy_args

    def with_new_field(self):
        a = legacy(self)
        a.new_setting = 1
        return a
    monkeypatch.setattr(Experiment, 'legacy_args', with_new_field)
    ran = []
    monkeypatch.setattr(train, 'main', lambda e, config=None, sources=None: ran.append(e))
    train.main_from_flags('--task se2c --model hybrid4c --size xs --note test'.split())
    assert len(ran) == 1


def test_old_se2_checkpoints_load_as_se2d(tmp_path):
    from mapping.checkpoint import load_checkpoint
    module = tasks.get('se2d')['models'].get('vanilla4')
    spec = tasks.config_and_spec('se2d', 32, 8, 0)[1]
    args = vars(train.parse('--task se2d --model vanilla4 --size xs'.split()))
    # exactly as written before the rename: no arch, no provenance
    torch.save({'model': module.build('xs', spec).state_dict(), 'args': {**args, 'task': 'se2'},
                'spec': spec.__dict__, 'task': 'se2'}, tmp_path / 'final.pt')
    assert load_checkpoint(tmp_path / 'final.pt')[3] == 'se2d'


def test_summarize_reads_every_task_and_the_old_name(tmp_path):
    import subprocess
    import sys
    run.main(['model=se2c/vanilla4c', f'run.out={tmp_path}/vanilla4c/xs/r', *TINY])
    for task in ('se2c', 'se2', 'se2d', 'z2'):
        r = subprocess.run([sys.executable, 'summarize.py', '--task', task, '--root',
                            str(tmp_path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    assert 'vanilla4c' in r.stdout


def test_a_branch_must_continue_its_parent(tmp_path):
    parent = ['model=se2c/vanilla4c', f'run.out={tmp_path}/p', 'run.save_every=1', *TINY]
    run.main(parent)
    ckpt = f'run.branch={tmp_path}/p/step1.pt'
    with pytest.raises(SystemExit, match='in task.episode_steps, task.generator.turn_n'):
        run.main(['model=se2c/vanilla4c', f'run.out={tmp_path}/c1', ckpt, *TINY,
                  'task.generator.turn_n=4', 'task.episode_steps=16'])
    run.main(['model=se2c/vanilla4c', f'run.out={tmp_path}/c2', ckpt, *TINY,
              'schedule=wsd', 'run.steps=3'])          # a new schedule and length: fine
    assert os.path.exists(tmp_path / 'c2' / 'final.json')


def _job_command(path, values):
    """The run.py command a job file runs, its shell variables filled."""
    import re
    import shlex
    text = open(path).read().replace('\\\n', ' ')
    line = next(ln for ln in text.splitlines() if ln.startswith('python run.py'))
    for k, v in values.items():
        line = line.replace(k, v)
    assert '$' not in line, line
    return shlex.split(line)[2:]


def test_job_generators_write_commands_that_compose(tmp_path):
    """scaling.py and tools/branch_jobs.py write run.py commands (with the
    dependency preflight) that compose and build."""
    import subprocess
    import sys
    here = os.getcwd()
    r = subprocess.run([sys.executable, os.path.join(here, 'scaling.py'), '--task', 'se2d',
                        '--model', 'hybrid4', '--slurm', '--sizes', 'xs', 's', '--lr-grid', '1e-3',
                        '--extra', 'task.query_times=uniform schedule=wsd'],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    job = tmp_path / 'jobs' / 'se2d_hybrid4.sbatch'
    assert 'python scripts/check_env.py || exit 1' in job.read_text()
    argv = _job_command(job, {'$SIZE': 's', '$RUN': 'seed0_lr1e-3', '$LRFLAG': 'optim.lr=1e-3'})
    e = run.build(run.resolved(run.load(argv)))
    assert (e.model.size, e.optim.lr, e.schedule.name, e.run.name, e.task.generator.workers) == (
        's', 1e-3, 'wsd', 'seed0_lr1e-3', 4)
    assert e.run.note == 'scaling se2d hybrid4 s seed0_lr1e-3'      # one note per task
    out = tmp_path / 'cd.sbatch'
    r = subprocess.run([sys.executable, os.path.join(here, 'tools', 'branch_jobs.py'),
                        '--task', 'se2c', '--model', 'hybrid4c', '--sizes', 'xs',
                        '--parent', 'p', '--budgets', '62500', '--out', str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    argv = _job_command(out, {'$SIZE': 'xs', '$PARENT/step$K.pt': 'results/se2c/x/step50000.pt',
                              '$(( S / 1000 ))': '62', '$S': '62500'})
    e = run.build(run.resolved(run.load(argv)))
    assert (e.task.name, e.run.steps, e.run.name, e.schedule.name) == ('se2c', 62500, 'p_cd62k', 'wsd')
    assert e.run.note == 'cooldown se2c hybrid4c xs p to 62500 steps'


def test_a_value_for_another_variant_is_explained(tmp_path):
    f = tmp_path / 'x.yaml'
    f.write_text('# @package _global_\ndefaults:\n  - override /schedule: wsd\n'
                 'schedule:\n  cooldown: 0.3\n')
    with pytest.raises(SystemExit, match='cooldown is set .* mapping.config.Cosine, has no such field'):
        run.main(['--exp', str(f), 'model=z2/hybrid4', 'schedule=cosine', '--print'])
