"""``run.init_from``: a weights-only warm start (docs/config.md).

A new run (own run id, fresh optimizer and schedule, step 0) whose model
starts from a checkpoint's weights. The architecture must match exactly;
task settings may differ; ``run.branch`` together with it is refused. The
parent checkpoint's path, sha256 and run id are recorded in config.yaml,
overrides.yaml and the ledger, and a ``--from`` rerun verifies them."""
import hashlib
import json

import pytest
import torch
from omegaconf import OmegaConf

import run
from tests.test_provenance import TINY


@pytest.fixture(scope='module')
def parent(tmp_path_factory):
    d = tmp_path_factory.mktemp('parent') / 'p'
    run.main([*TINY, f'run.out={d}', 'run.save_every=1', 'run.note=parent'])
    return d


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def test_weights_equal_the_checkpoint_and_the_optimizer_is_fresh(parent, tmp_path):
    ck = parent / 'step1.pt'
    run.main([*TINY, f'run.out={tmp_path}/c', f'run.init_from={ck}', 'optim.lr=0',
              'run.save_every=1', 'run.note=child'])
    p = torch.load(ck, weights_only=False)
    c = torch.load(tmp_path / 'c' / 'final.pt', weights_only=False)
    # lr 0: no step moves a weight (AdamW's decay is scaled by lr too), so the
    # final weights are the step-0 weights, which must be the parent's
    assert p['model'].keys() == c['model'].keys()
    assert all(torch.equal(p['model'][k], c['model'][k]) for k in p['model'])
    # a fresh optimizer: after this run's first step Adam's per-parameter
    # step count is 1, not the parent's 1 + 1
    c1 = torch.load(tmp_path / 'c' / 'step1.pt', weights_only=False)
    assert {int(s['step']) for s in c1['opt']['state'].values()} == {1}
    assert c1['step'] == 1
    first = json.loads(open(tmp_path / 'c' / 'log.jsonl').readline())
    assert first['start'] == 0 and first['branch'] is None
    assert first['init_from']['sha256'] == sha(ck)


def test_the_parent_is_recorded_and_a_rerun_verifies_it(parent, tmp_path):
    ck = parent / 'step1.pt'
    run.main([*TINY, f'run.out={tmp_path}/c', f'run.init_from={ck}', 'run.note=child'])
    parent_id = (parent / 'RUN_ID').read_text().strip()
    child_id = (tmp_path / 'c' / 'RUN_ID').read_text().strip()
    assert child_id != parent_id
    led = json.load(open(tmp_path / 'c' / 'provenance.json'))
    i = led['parent']['init_from']
    assert i['run_id'] == parent_id and i['sha256'] == sha(ck) and i['step'] == 1
    assert i['path'] == str(ck) and 'checkpoint stamp' in i['run_id_evidence']
    assert led['parent']['branch_from'] is None
    cfg = OmegaConf.load(tmp_path / 'c' / 'config.yaml').run
    assert (cfg.init_from, cfg.init_sha256, cfg.init_run_id) == (str(ck), sha(ck), parent_id)
    final = json.load(open(tmp_path / 'c' / 'final.json'))
    assert final['init_from']['run_id'] == parent_id and final['args']['init_from'] == str(ck)
    # a rerun from the saved config verifies the same checkpoint ...
    run.main(['--from', str(tmp_path / 'c' / 'config.yaml'), f'run.out={tmp_path}/r',
              'run.note=again'])
    r = json.load(open(tmp_path / 'r' / 'provenance.json'))['parent']['init_from']
    assert (r['sha256'], r['run_id']) == (i['sha256'], i['run_id'])
    # ... and refuses one that changed
    with pytest.raises(SystemExit, match='sha256'):
        run.main([*TINY, f'run.out={tmp_path}/x', f'run.init_from={ck}',
                  f'run.init_sha256={"0" * 63 + "1"}', 'run.note=x'])


def test_task_settings_may_differ(parent, tmp_path):
    run.main([*TINY, f'run.out={tmp_path}/c', f'run.init_from={parent / "final.pt"}',
              'task.episode_steps=16', 'task.generator.path=saw', 'run.note=child'])
    f = json.load(open(tmp_path / 'c' / 'final.json'))
    assert f['args']['episode_steps'] == 16 and f['path'] == 'saw'


def test_an_architecture_mismatch_is_refused(parent, tmp_path):
    with pytest.raises(SystemExit, match='architecture mismatch'):
        run.main([*TINY, 'model.size=s', f'run.out={tmp_path}/c',
                  f'run.init_from={parent / "final.pt"}', 'run.note=child'])
    assert not (tmp_path / 'c').exists()


def test_init_from_and_branch_together_are_refused(parent, tmp_path):
    with pytest.raises(SystemExit, match='init_from and run.branch'):
        run.main([*TINY, f'run.out={tmp_path}/c', f'run.init_from={parent / "final.pt"}',
                  f'run.branch={parent / "step1.pt"}', 'run.steps=4', 'run.note=child'])


def test_the_record_fields_need_init_from():
    with pytest.raises((Exception, SystemExit), match='without'):
        run.build(run.resolved(run.load([*TINY, f'run.init_sha256={"a" * 64}', 'run.note=x'])))
