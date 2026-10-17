"""Dependency preflight: does this Python environment have everything the code imports?

    python scripts/check_env.py            # the training code (requirements.txt)
    python scripts/check_env.py --dev      # plus tools/, tests/, scripts/, experiments/

Three checks, all of which must pass:
1. **Coverage (static).** Every third-party module imported anywhere in the
   checked code (``import``/``from`` at any depth, lazy imports included) is
   named in the requirements file. Catches code that grew a new dependency
   without declaring it (``mapping/ocean/seen.py`` importing scipy, 2026-10-17).
2. **Installed.** Every requirement is installed, and ``==`` pins match.
3. **Importable.** Every third-party module the code imports can be found.

Exit status 0 when all pass, 1 otherwise, with the exact ``pip install`` line to
run. Every sbatch job runs this first, and ``jobs/submit.sh`` runs it on the
login node before ``sbatch``, so a missing package fails in seconds rather than
after the queue wait (CONTRIBUTING: Torch workflow).
"""
import argparse
import ast
import importlib.metadata as md
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNTIME = ['mapping', 'run.py', 'train.py', 'render.py', 'plot_curves.py']
DEV = ['tools', 'tests', 'scripts', 'experiments']
SKIP_PARTS = {'shelved', 'quarantine', 'results', 'legacy', '__pycache__'}
# import name -> distribution name, where they differ
DIST = {'PIL': 'pillow', 'yaml': 'PyYAML', 'fla': 'fla-core', 'hydra': 'hydra-core'}


def python_files(paths):
    for rel in paths:
        p = ROOT / rel
        files = [p] if p.is_file() else sorted(p.rglob('*.py')) if p.is_dir() else []
        for f in files:
            if not SKIP_PARTS & set(f.relative_to(ROOT).parts):
                yield f


def local_names():
    """Top-level names that resolve inside the repository (packages, modules,
    and sibling modules that scripts import from their own directory)."""
    names = set()
    for f in ROOT.rglob('*.py'):
        if SKIP_PARTS & set(f.relative_to(ROOT).parts):
            continue
        names.add(f.stem)
        names.update(part for part in f.relative_to(ROOT).parts[:-1])
    return names


def imports(paths):
    """{top-level third-party module: [files importing it]}."""
    local, found = local_names(), {}
    for f in python_files(paths):
        try:
            tree = ast.parse(f.read_text(), str(f))
        except SyntaxError as e:
            raise SystemExit(f'check_env: cannot parse {f}: {e}')
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [n.module]
            else:
                continue
            for m in mods:
                top = m.split('.')[0]
                if top in sys.stdlib_module_names or top in local or top == '__future__':
                    continue
                found.setdefault(top, []).append(str(f.relative_to(ROOT)))
    return found


def requirements(path):
    """{normalised dist name: (pin or None, line)} from a requirements file,
    following ``-r`` includes."""
    reqs = {}
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('-r'):
            reqs.update(requirements(path.parent / line[2:].strip()))
            continue
        m = re.match(r'^([A-Za-z0-9_.\-]+)\s*(==\s*([^\s;]+))?', line)
        if not m:
            raise SystemExit(f'check_env: cannot read {path.name} line {line!r}')
        reqs[norm(m.group(1))] = (m.group(3), line)
    return reqs


def norm(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def check(dev=False, verbose=True):
    req_file = ROOT / ('requirements-dev.txt' if dev else 'requirements.txt')
    reqs = requirements(req_file)
    used = imports(RUNTIME + (DEV if dev else []))
    problems = []
    for mod, files in sorted(used.items()):
        if norm(DIST.get(mod, mod)) not in reqs:
            problems.append(f'undeclared: {mod} (imported by {", ".join(sorted(set(files))[:3])}) '
                            f'is not in {req_file.name}; add {DIST.get(mod, mod)}')
    missing = []
    for dist, (pin, line) in sorted(reqs.items()):
        try:
            have = md.version(dist)
        except md.PackageNotFoundError:
            missing.append(line)
            problems.append(f'not installed: {line}')
            continue
        if pin and have != pin:
            missing.append(line)
            problems.append(f'version: {dist} {have} installed, {req_file.name} pins {pin}')
    for mod in sorted(used):
        if importlib.util.find_spec(mod) is None and norm(DIST.get(mod, mod)) in reqs:
            problems.append(f'not importable: {mod}')
    if verbose:
        print(f'check_env: {sys.executable}, {req_file.name}, {len(used)} third-party modules, '
              f'{len(reqs)} requirements: {"OK" if not problems else f"{len(problems)} problem(s)"}')
        for p in problems:
            print(f'  {p}')
        if missing:
            print('  fix: pip install ' + ' '.join(repr(m) if ' ' in m else m for m in missing))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dev', action='store_true', help='also tools/, tests/, scripts/ and experiments/ (requirements-dev.txt)')
    args = ap.parse_args()
    sys.exit(1 if check(dev=args.dev) else 0)


if __name__ == '__main__':
    main()
