"""Every third-party import is declared (scripts/check_env.py, coverage check).

Static, so it runs anywhere: adding ``import scipy`` to mapping/ without a
requirements.txt line fails here, before a Torch job can fail on it."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('check_env', ROOT / 'scripts' / 'check_env.py')
check_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_env)


def undeclared(dev):
    reqs = check_env.requirements(ROOT / ('requirements-dev.txt' if dev else 'requirements.txt'))
    used = check_env.imports(check_env.RUNTIME + (check_env.DEV if dev else []))
    return sorted(m for m in used if check_env.norm(check_env.DIST.get(m, m)) not in reqs)


def test_training_imports_are_in_requirements():
    assert undeclared(dev=False) == []


def test_dev_imports_are_in_requirements_dev():
    assert undeclared(dev=True) == []


def test_coverage_check_catches_an_undeclared_import(tmp_path, monkeypatch):
    pkg = tmp_path / 'mapping'
    pkg.mkdir()
    (pkg / 'new.py').write_text('def f():\n    import scipy.ndimage\n    import json, mapping.x\n')
    (tmp_path / 'requirements.txt').write_text('numpy\n')
    monkeypatch.setattr(check_env, 'ROOT', tmp_path)
    used = check_env.imports(['mapping'])
    assert sorted(used) == ['scipy']                  # stdlib and local names skipped
    reqs = check_env.requirements(tmp_path / 'requirements.txt')
    assert check_env.norm('scipy') not in reqs


def test_requirements_follow_includes_and_pins(tmp_path):
    (tmp_path / 'a.txt').write_text('hydra-core==1.3.5  # pinned\nPyYAML\n')
    (tmp_path / 'b.txt').write_text('-r a.txt\npytest\n')
    reqs = check_env.requirements(tmp_path / 'b.txt')
    assert reqs['hydra-core'][0] == '1.3.5' and reqs['pyyaml'][0] is None and 'pytest' in reqs

