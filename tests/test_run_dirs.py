"""Run directories: one template for every task (docs/config.md, "Run
directories"; provenance design §4 and §14).

    run.dir = ${run.root}/${task.name}/${model.name}/${model.size}/<run.name or seed<seed>>

``run.out`` is the explicit override. Nothing derives a run's directory
from a per-task root any more (``mapping/tasks.py`` has none); readers use
``tasks.results_dir``. z2 runs live under results/z2/ like every other
task; the old ``results/<model>/`` layout is read nowhere, and
``scripts/sync.sh pull`` maps a remote tree that still has it into
results/z2/. The comparison of these paths against the pre-merge
reference code is in tests/test_equivalence_runs.py.
"""
import os
import shutil
import subprocess

import pytest
from omegaconf import OmegaConf

import mapping.tasks as tasks
import run

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = ['run.device=cpu', 'run.steps=2', 'eval.every=2', 'eval.log_every=2',
        'eval.batches=1', 'optim.batch=2', 'run.note=test']


def location(*overrides):
    return run.build(run.resolved(run.load(list(overrides)))).run.location()


@pytest.mark.parametrize('model,task', [('z2/vanilla4', 'z2'), ('se2d/hybrid4', 'se2d'),
                                        ('se2c/vanilla4c', 'se2c')])
def test_one_template_for_every_task(model, task):
    name = model.split('/')[1]
    assert location(f'model={model}') == {
        'dir': f'results/{task}/{name}/xs/seed0', 'root': 'results',
        'rel': f'{task}/{name}/xs/seed0'}
    assert location(f'model={model}', 'run.seed=3')['dir'] == f'results/{task}/{name}/xs/seed3'
    assert location(f'model={model}', 'model.size=s', 'run.name=a_b')['dir'] == \
        f'results/{task}/{name}/s/a_b'
    assert location(f'model={model}', 'run.root=/x/y')['dir'] == f'/x/y/{task}/{name}/xs/seed0'
    assert tasks.results_dir(task) == f'results/{task}'


def test_out_wins_and_a_preset_may_restructure():
    assert location('model=z2/vanilla4', 'run.out=/tmp/o') == {
        'dir': '/tmp/o', 'root': 'results', 'rel': None}
    loc = location('model=se2d/hybrid4', 'run.dir=${run.root}/campaigns/h1c/${model.name}')
    assert loc == {'dir': 'results/campaigns/h1c/hybrid4', 'root': 'results',
                   'rel': 'campaigns/h1c/hybrid4'}


def test_no_task_carries_a_root_and_the_alias_keeps_its_old_directory():
    assert all('root' not in row for row in tasks.ALL.values())
    assert tasks.results_dir('se2') == 'results/se2'        # old runs, not moved
    assert tasks.get('se2') is tasks.get('se2d')
    with pytest.raises(KeyError):
        tasks.results_dir('nope')


def test_a_run_writes_to_its_template_directory(tmp_path):
    run.main(['model=z2/vanilla4', 'task.generator.workers=0', f'run.root={tmp_path}',
              'run.name=t', *TINY])
    d = tmp_path / 'z2' / 'vanilla4' / 'xs' / 't'
    assert (d / 'final.json').exists()
    saved = OmegaConf.load(d / 'config.yaml')
    assert (saved.run.root, saved.run.dir) == (str(tmp_path), str(d))
    assert sorted(os.listdir(tmp_path)) == ['z2']
    assert (d / 'RUN_ID').exists() and (d / 'provenance.json').exists()


@pytest.mark.skipif(shutil.which('rsync') is None, reason='needs rsync')
def test_sync_pull_maps_the_old_z2_layout_into_results_z2(tmp_path):
    remote, here = tmp_path / 'remote', tmp_path / 'here'
    for p in ('hybrid/xs/r', 'vanilla_depth/d4/seed0', 'se2/hybrid4/xs/r', 'se2d/h/s/r',
              'se2c/x/m/r', 'z2/lstm/xs/r'):
        os.makedirs(remote / 'results' / p)
        (remote / 'results' / p / 'final.json').write_text('{}')
    (remote / 'results' / 'curves.png').write_text('')
    os.makedirs(here / 'scripts')
    shutil.copy(os.path.join(HERE, 'scripts', 'sync.sh'), here / 'scripts')
    r = subprocess.run(['bash', str(here / 'scripts' / 'sync.sh'), 'pull'],
                       env={**os.environ, 'MAPPING_REMOTE': str(remote)},
                       capture_output=True, text=True)
    got = sorted(os.path.relpath(os.path.join(dp, f), here / 'results')
                 for dp, _, fs in os.walk(here / 'results') for f in fs)
    assert got == ['se2/hybrid4/xs/r/final.json', 'se2c/x/m/r/final.json',
                   'se2d/h/s/r/final.json', 'z2/curves.png', 'z2/hybrid/xs/r/final.json',
                   'z2/lstm/xs/r/final.json', 'z2/vanilla_depth/d4/seed0/final.json'], r.stderr
