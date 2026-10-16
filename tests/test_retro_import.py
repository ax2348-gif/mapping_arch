"""The retroactive ledger records (tools/import_retro.py; provenance design
§11, the user's decisions in §14).

Only MAJOR experiments are imported (experiments/registry.yaml,
``experiments:``, with the borderline calls: pre-freeze ablations and the
time-recurrence pair major, the blind probe not). Every record is marked
retroactive and claims no more than the audit found: no snapshot, no
``refs/runs/`` ref, the audit's commit string verbatim, dirty unknown.
"""
import csv
import glob
import json
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(HERE, 'experiments', 'ledger')


def retro():
    return {r['audit']['row']: r for r in (json.load(open(f)) for f in
                                           glob.glob(os.path.join(LEDGER, '*', 'legacy-*.json')))}


def test_exactly_the_major_runs_are_imported():
    reg = yaml.safe_load(open(os.path.join(HERE, 'experiments', 'registry.yaml')))
    major = {r['id']: e for e in reg['experiments'] if e['status'] == 'major' for r in e['runs']}
    ids = {e['id'] for e in reg['experiments'] if e['status'] == 'major'}
    assert {'se2d-prefreeze-ablations', 'se2d-time-recurrence',
            'se2d-hybrid-original-sweep'} <= ids                   # the user's calls
    assert 'lift-blind-probe' not in ids
    recs = retro()
    assert set(recs) == set(major)
    not_major = {r for b in reg['borderline'] for r in b['runs']}
    assert not not_major & set(recs)
    for row, rec in recs.items():
        assert rec['run']['note'] == major[row]['title']
        assert rec['run']['tags'] == ['major', major[row]['id']]


def test_retro_records_claim_nothing_the_audit_did_not_find():
    audit = {r['run_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'experiments',
                                                                      'audit', 'runs.csv')))}
    refs = subprocess.run(['git', 'for-each-ref', '--format=%(refname)', 'refs/runs/legacy-*'],
                          cwd=HERE, capture_output=True, text=True).stdout
    assert refs == ''
    for row, rec in retro().items():
        a = audit[row]
        c = rec['code']
        assert rec['retroactive'] is True and rec['schema'] == 1
        assert (c['snapshot'], c['ref'], c['dirty']) == (None, None, None)
        assert c['commit'] == (a['commit'] or 'unknown')
        if c['commit'].startswith('inferred:'):
            assert c['recorded'] is None and c['inferred'] == c['commit'][9:]
        elif c['commit'] == 'unknown':
            assert c['recorded'] is None and c['inferred'] is None and c['resolved'] is None
        else:
            assert c['recorded'] == c['commit'] and 'recorded' in c['basis']
        if c['tree_verified']:
            assert a['tree'].startswith('/scratch/ax2348/mapping_hydra') and c['recorded'] == '50341e4'
        assert rec['status_as_found'] == a['status']
        assert rec['created_as_found'] == (a['start'] or None)
        assert rec['progress']['last_step'] == (int(a['last_step']) if a['last_step'] else None)
        assert rec['host'] == row.split(':')[0]


def test_one_file_per_record_named_by_its_id_under_its_host():
    for f in glob.glob(os.path.join(LEDGER, '*', 'legacy-*.json')):
        rec = json.load(open(f))
        assert os.path.basename(f) == rec['run_id'] + '.json'
        assert os.path.basename(os.path.dirname(f)) == rec['host']
