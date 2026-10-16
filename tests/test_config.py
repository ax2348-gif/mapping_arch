"""The Hydra config tree (conf/, mapping/config.py, run.py).

1. Every field is consumed. Every composable config (each model file, on
   each schedule, and each experiment preset) instantiates; an unknown key
   at any node is an error; and changing any value changes what the
   trainer receives (the settings, or for architecture fields the built
   model). So no field is ever composed that nothing reads.
2. The size ladders moved into conf/model/size/ are the old in-code
   tables, entry for entry and in order (tests/sizes_snapshot.json).
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


CONSUMED_CASES = [
    ['model=z2/hybrid4'], ['model=z2/vanilla', 'schedule=wsd'], ['model=z2/lstm'],
    ['model=z2/vanilla_depth'], ['model=se2d/vanilla4', 'schedule=wsd'],
    ['model=se2d/hybrid45b_two'], ['model=se2c/hybrid4c'],
    ['+experiment=se2c_bridge'], ['+experiment=se2d_scaling_wsd'],
]


@pytest.mark.parametrize('overrides', CONSUMED_CASES, ids=lambda o: ' '.join(o))
def test_every_field_is_consumed(overrides):
    """Change each leaf in turn: the settings the trainer receives must
    change, or, for an architecture field, the model must."""
    base = OmegaConf.to_container(compose(*overrides))
    exp = run.build(OmegaConf.create(base))
    a0 = vars(exp.legacy_args())
    spec = tasks.config_and_spec(a0['task'], 32, 8, 0, 'uniform')[1]
    for path, value in leaves(base):
        cfg = copy.deepcopy(base)
        node = cfg
        for k in path[:-1]:
            node = node[k]
        if path[0] == 'model' and path[1] == 'arch':
            # an architecture field: it must reach the model's constructor
            node[path[-1]] = (value * 2 if isinstance(value, int) else
                              (['first'] if value != ['first'] else ['last']))
            m = run.build(OmegaConf.create(cfg)).model
            assert m.arch.keywords[path[-1]] == (tuple(node[path[-1]])
                                                 if isinstance(node[path[-1]], list)
                                                 else node[path[-1]]), path
            continue
        node[path[-1]] = _bump(value)
        try:
            a = vars(run.build(OmegaConf.create(cfg)).legacy_args())
        except (InstantiationException, KeyError, ValueError, TypeError):
            continue          # rejected outright (an unknown task or model name): consumed
        assert a != a0, f'{".".join(path)} changed but the run settings did not'


@pytest.mark.parametrize('node', ['', 'optim', 'run', 'eval', 'schedule', 'task',
                                  'task.generator', 'model', 'model.arch'])
def test_unknown_keys_are_errors(node):
    cfg = compose('model=se2d/hybrid4')
    target = OmegaConf.select(cfg, node) if node else cfg
    target['not_a_field'] = 1        # composed configs are saved unlocked
    with pytest.raises(Exception, match='not_a_field'):
        run.build(cfg)


def test_ladders_are_the_old_tables():
    for task in ('z2', 'se2d', 'se2c'):
        for name, module in tasks.get(task)['models'].REGISTRY.items():
            assert [[k, v] for k, v in module.SIZES.items()] == SNAPSHOT[f'{task}/{name}'], name


def test_size_files_match_the_ladders():
    """conf/model/<task>/<name>.yaml at each size gives the module's SIZES
    entry, and its size name."""
    for model in MODELS:
        task, name = model.split('/')
        module = tasks.get(task)['models'].get(name)
        default = OmegaConf.load(f'conf/model/{model}.yaml').defaults[1]['/model/size']
        prefix = default.rsplit('/', 1)[0] + '/' if '/' in default else ''
        for size, kwargs in module.SIZES.items():
            m = run.build(compose(f'model={model}', f'model/size={prefix}{size}')).model
            assert m.size == size
            arch = {k: v for k, v in m.arch.keywords.items() if k != 'placement'}
            assert arch == kwargs, (model, size)


@pytest.mark.parametrize('model', MODELS)
def test_config_models_are_the_old_models(model):
    """Same seed, same parameters, bit for bit, at every size."""
    task, name = model.split('/')
    module = tasks.get(task)['models'].get(name)
    spec = tasks.config_and_spec(task, 32, 8, 0, 'uniform')[1]
    default = OmegaConf.load(f'conf/model/{model}.yaml').defaults[1]['/model/size']
    prefix = default.rsplit('/', 1)[0] + '/' if '/' in default else ''
    for size in module.SIZES:
        torch.manual_seed(0)
        old = module.build(size, spec).state_dict()
        torch.manual_seed(0)
        new = run.build(compose(f'model={model}', f'model/size={prefix}{size}')).model.build(spec)
        new = new.state_dict()
        assert old.keys() == new.keys(), (model, size)
        assert all(torch.equal(old[k], new[k]) for k in old), (model, size)
        assert type(module.build(size, spec)) is type(run.build(
            compose(f'model={model}', f'model/size={prefix}{size}')).model.build(spec))


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
    '--model hybrid4 --size xs --out /tmp/a\\ b --branch x=y.pt --cooldown 0.2',
]


@pytest.mark.parametrize('line', FLAG_CASES)
def test_flags_and_config_agree(line):
    import shlex
    parsed = train.parse(shlex.split(line))
    args = run.build(run.resolved(run.load(train.overrides_from_flags(parsed)))).legacy_args()
    if parsed.lr is None:
        parsed.lr = model_lr(parsed.task, parsed.model, parsed.size)
    assert vars(args) == vars(parsed)


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
                   '  - override /model/size: m\n  - override /schedule: wsd\n'
                   'run:\n  steps: 123\ntask:\n  generator:\n    texture: imagenette\n')
    a = run.build(run.resolved(run.load(['run.seed=3'], exp=str(exp)))).legacy_args()
    assert (a.task, a.model, a.size, a.schedule, a.steps, a.seed, a.texture) == (
        'se2d', 'vanilla4', 'm', 'wsd', 123, 3, 'imagenette')
    bad = tmp_path / 'typo.yaml'
    bad.write_text('# @package _global_\nrun:\n  stpes: 5\n')
    with pytest.raises(Exception, match='stpes'):
        run.build(run.resolved(run.load(['model=z2/hybrid4'], exp=str(bad))))


def test_saved_config_reruns(tmp_path):
    cfg = compose('+experiment=se2c_bridge', 'model/size=s')
    path = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, path)
    loaded = OmegaConf.load(path)
    assert vars(run.build(loaded).legacy_args()) == vars(run.build(cfg).legacy_args())
    loaded.merge_with(OmegaConf.from_dotlist(['run.seed=7']))
    assert run.build(loaded).legacy_args().seed == 7
