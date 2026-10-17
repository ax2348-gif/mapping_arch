"""Train one run from the Hydra config tree (``conf/``, docs/config.md).

    python run.py model=se2d/hybrid4 model.size=s run.steps=500
    python run.py +experiment=se2c_bridge model.size=m
    python run.py --exp path/to/my_experiment.yaml run.seed=1
    python run.py --from results/se2c/hybrid4c/m/<run>/config.yaml run.name=rerun
    python run.py ... --print            # the resolved config and train.py flags; no run

Every run needs ``run.note='<what this run is for>'`` on the command line,
like a commit message (``mapping.provenance.check_note``). Each run is
pinned to a snapshot commit of its code (``refs/runs/<run_id>``) and gets
a ledger record (``experiments/ledger/<host>/<run_id>.json``) before it
trains (mapping/provenance.py).

Arguments are Hydra overrides (``key=value``, ``group=option``,
``+experiment=name``). ``--exp FILE`` composes an experiment file from
anywhere exactly like a preset in ``conf/experiment/``. ``--from FILE``
reruns a run from its saved ``config.yaml``: recomposed from the sources
saved beside it plus the new overrides (``rerun``). A run directory that
is not empty is refused (``run.overwrite=true`` to replace it), so a
rerun needs a new ``run.name``.

The tree is composed with Hydra's compose API rather than ``@hydra.main``,
whose command-line parser fails on Python 3.14 ("badly formed help
string") and which would change the working directory and logging.
"""
import hashlib
import os
import re
import shutil
import sys
import tempfile

from hydra import compose, initialize_config_dir
from hydra.errors import (ConfigCompositionException, InstantiationException,
                          MissingConfigException)
from hydra.utils import instantiate
from omegaconf import OmegaConf

import mapping.config  # noqa: F401  (registers the model_lr resolver)

CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf')


def models():
    """Every model choice, ``<task>/<name>``."""
    import glob
    root = os.path.join(CONF, 'model')
    return sorted(os.path.relpath(f, root)[:-5] for f in glob.glob(os.path.join(root, '*', '*.yaml')))


def explain(error, overrides):
    """A message saying what to do, for Hydra composition errors a user
    meets; None for the rest."""
    msg = str(error)
    if "You must specify 'model'" in msg:
        return 'choose a model: model=<task>/<name>, one of ' + ', '.join(models())
    m = re.search(r"Could not find 'model/([^']+)'", msg)
    if m:
        return (f'no model {m.group(1)!r} (conf/model/{m.group(1)}.yaml); have '
                + ', '.join(models()))
    m = re.search(r"Could not override '([^']+)'", msg)
    if m:
        key = m.group(1)
        if key.startswith(('model.arch.', 'model.size_kwargs.')):
            return (f'{key}: a model\'s architecture is its name at a rung of its ladder. '
                    f'Pick a rung with model.size=<rung> (conf/ladders/<ladder>.yaml), add '
                    f'a rung there, or register a new model (docs/config.md, "How to add")')
        parent, _, leaf = key.rpartition('.')
        try:
            node = OmegaConf.select(load([o for o in overrides if _key(o) != key]), parent)
            have = [k for k in node if k not in ('_target_', '_partial_')]
        except Exception:
            have = None
        return (f'{key}: {parent or "the config"} has no field {leaf!r}'
                + (f'; its fields: {", ".join(have)}' if have else '')
                + '. (Hydra\'s +key=value would add one, which the strict objects then refuse.)')
    m = re.search(r"Ambiguous value for argument '([^']+)'", msg)
    if m:
        return (f'{m.group(1)}: run.py launches one run. For a sweep, launch one run per '
                f'value with a distinct run.name (e.g. "run.name=lr${{optim.lr}}"), or let '
                f'scaling.py / tools/branch_jobs.py write the job array (docs/config.md, '
                f'"Sweeps"). A list value is key=[a,b]')
    return None


def load(overrides=(), exp=None):
    """The composed config: ``conf/config.yaml`` with ``overrides``, and
    the experiment file ``exp`` (any path) composed as an experiment
    preset. It is copied under a name of its own (``_exp_<hash>``), so a
    file named like a preset in ``conf/experiment/`` is still this file."""
    overrides = list(overrides)
    with tempfile.TemporaryDirectory() as extra:
        if exp is not None:
            if not os.path.exists(exp):
                raise SystemExit(f'--exp {exp}: no such file')
            with open(exp) as f:
                text = f.read()
            if '@package _global_' not in text:
                raise SystemExit(f'--exp {exp}: an experiment file starts with the line '
                                 f'"# @package _global_" (its keys are whole-config '
                                 f'paths: run.steps, task.generator...); see docs/config.md')
            os.makedirs(os.path.join(extra, 'experiment'))
            name = '_exp_' + hashlib.sha256(text.encode()).hexdigest()[:12]
            shutil.copy(exp, os.path.join(extra, 'experiment', name + '.yaml'))
            overrides = [f'hydra.searchpath=[file://{extra}]',
                         f'+experiment={name}'] + overrides
        try:
            with initialize_config_dir(version_base='1.3', config_dir=CONF):
                return compose('config', overrides=overrides)
        except (ConfigCompositionException, MissingConfigException) as e:
            error = e
    message = explain(error, [o for o in overrides if not o.startswith('hydra.')])
    if message is None:
        raise error
    raise SystemExit(message) from error


def resolved(cfg):
    """The config with every interpolation resolved, as runs save it."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def build(cfg):
    """The ``mapping.config.Experiment`` a config describes."""
    return instantiate(cfg, _convert_='all')


def _key(override):
    return override.split('=', 1)[0].lstrip('+~')


def merge_overrides(saved, new):
    """``saved`` with ``new`` applied: a new override replaces a saved one
    of the same key."""
    keys = {_key(o) for o in new}
    return [o for o in saved if _key(o) not in keys] + list(new)


def differences(a, b, prefix=''):
    """Dotted keys whose values differ between two plain dicts."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, '<missing>'), b.get(k, '<missing>')
        if isinstance(x, dict) and isinstance(y, dict):
            out += differences(x, y, f'{prefix}{k}.')
        elif x != y:
            out.append(f'{prefix}{k}')
    return out


# Fields added after runs were saved, so absent from older config.yaml
# files: the run directory template (2026-10-16), which says where a run is
# written, never what it trains; and model.query_mode (2026-10-17), whose
# default (mixed) is what every run before it trained. Recomposing an older
# run may add them (the current defaults) without that counting as drift.
ADDED_RUN_FIELDS = ('run.root', 'run.dir', 'run.note', 'run.tag', 'run.provenance',
                    'model.query_mode')   # 2026-10-17; absent = mixed, its default

# Run fields that are metadata about a launch, not settings: never compared
# as drift, never carried into a rerun's sources (a rerun gets its own note)
LAUNCH_ONLY = ('run.overwrite', 'run.note')
METADATA = ('run.note', 'run.tag', 'run.provenance')


def _added_since(saved):
    return [k for k in ADDED_RUN_FIELDS
            if OmegaConf.select(OmegaConf.create(saved), k, default='<missing>') == '<missing>']


def rerun(frm, overrides):
    """``(config, sources)`` to rerun the run whose ``config.yaml`` is
    ``frm``, with ``overrides`` applied.

    A run saved with its sources (``overrides.yaml`` beside it, and its
    experiment file as ``experiment.yaml``) is recomposed from them plus
    ``overrides``, so group choices (``schedule=wsd``) and sizes
    (``model.size=s``) apply and derived values (the rung's arguments, the
    model's learning rate) follow. Recomposing it unchanged must give its
    saved config: if conf/ has changed since, the rerun is refused rather
    than silently becoming another run.

    An older run (no ``overrides.yaml``) can only take value overrides of
    existing leaves, merged into its saved config."""
    d = os.path.dirname(os.path.abspath(frm))
    saved = OmegaConf.to_container(OmegaConf.load(frm))
    src = os.path.join(d, 'overrides.yaml')
    if os.path.exists(src):
        rec = OmegaConf.to_container(OmegaConf.load(src))
        exp = os.path.join(d, rec['experiment']) if rec.get('experiment') else None
        again = OmegaConf.to_container(resolved(load(rec['overrides'], exp)))
        drift = [k for k in differences(again, saved)
                 if k not in LAUNCH_ONLY and k not in _added_since(saved)]
        if drift:
            raise SystemExit(f'{frm}: composing its sources ({src}) with the current '
                             f'conf/ no longer gives its saved config; changed: '
                             f'{", ".join(drift)}. conf/ (or a ladder, or a model\'s '
                             f'default lr) changed since the run. Rerun it from the '
                             f'commit it was made at, or write a new experiment (--exp)')
        merged = merge_overrides(rec['overrides'], overrides)
        return resolved(load(merged, exp)), {'overrides': merged, 'experiment': exp}
    added = _added_since(saved)
    cfg = OmegaConf.create(saved)
    if added:                                # saved before the run.dir template
        base = OmegaConf.to_container(OmegaConf.load(os.path.join(CONF, 'config.yaml')))
        for k in added:                      # the template itself, resolved below
            node = base
            for part in k.split('.'):
                node = node[part]
            OmegaConf.update(cfg, k, node, force_add=True)
    for o in overrides:
        key = _key(o)
        if '=' not in o or o[0] in '+~' or '/' in key:
            raise SystemExit(f'{o}: {frm} has no overrides.yaml (it predates them), so '
                             f'only value overrides of existing keys apply (key=value), '
                             f'not group choices or added/removed keys')
        if key == 'run.note':                 # the launcher's, never the saved one
            continue
        if OmegaConf.select(cfg, key, default='<missing>') == '<missing>':
            raise SystemExit(f'{o}: {frm} has no key {key!r}')
        if OmegaConf.is_dict(OmegaConf.select(cfg, key)):
            raise SystemExit(f'{o}: {key} is a config node, not a value. {frm} has no '
                             f'overrides.yaml (it predates them), so a group choice '
                             f'cannot be recomposed; only key=value on existing leaves')
    cfg.run.note = None                      # a rerun's note is its own
    cfg.merge_with(OmegaConf.from_dotlist(overrides))
    if OmegaConf.select(cfg, 'run.overwrite', default=None) is None:
        cfg.run.overwrite = False            # runs saved before the field existed
    return resolved(cfg), {'overrides': list(overrides), 'experiment': None}


def flags_equivalent(cfg, experiment):
    """The ``train.py`` command that runs exactly ``cfg``, or a note saying
    there is none: its flags are translated back into a config, which must
    equal ``cfg`` (a setting or choice with no flag, a new schedule say,
    has none)."""
    import contextlib
    import io
    import train
    flags = train.flags(experiment.legacy_args())
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            again = resolved(load(train.overrides_from_flags(train.parse(flags))))
        drift = [k for k in differences(OmegaConf.to_container(again),
                                        OmegaConf.to_container(cfg))
                 if k != 'run.overwrite' and not k.startswith(METADATA)]
    except (SystemExit, Exception) as e:
        drift = [f'the flags do not parse ({e})']
    if drift:
        return '# no exact train.py equivalent (differs: ' + ', '.join(drift) + ')'
    note = OmegaConf.select(cfg, 'run.note')
    return 'train.py ' + ' '.join(flags) + (f' --note {train._value(note)}' if note else '')


def parse_cli(argv):
    """(overrides, exp file, from file, print?) from run.py's arguments."""
    argv, exp, frm = list(argv), None, None
    show = '--print' in argv
    argv = [a for a in argv if a != '--print']
    for flag in ('--exp', '--from'):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                raise SystemExit(f'{flag} needs a file')
            value = argv[i + 1]
            del argv[i:i + 2]
            if flag == '--exp':
                exp = value
            else:
                frm = value
    if exp and frm:
        raise SystemExit('--exp and --from do not combine')
    for a in argv:
        if a.split('=', 1)[0].lstrip('+~') == 'model/size':
            raise SystemExit(f'{a}: sizes are no longer a config group. A model file names '
                             f'its ladder (conf/ladders/); pick a rung with model.size=<rung>, '
                             f'e.g. model.size={a.split("=", 1)[-1].rsplit("/", 1)[-1]}')
    return argv, exp, frm, show


def main(argv=None):
    import train
    overrides, exp, frm, show = parse_cli(sys.argv[1:] if argv is None else argv)
    if frm is not None:
        cfg, sources = rerun(frm, overrides)
    else:
        cfg = resolved(load(overrides, exp))
        sources = {'overrides': overrides, 'experiment': exp}
    try:
        experiment = build(cfg)
    except InstantiationException as e:
        m = re.search(r"target '([\w.]+)'.*unexpected keyword argument '(\w+)'", str(e), re.S)
        hint = (f'\n{m.group(2)} is set (by a file in the composition, the experiment or an '
                f'override) for a node whose class, {m.group(1)}, has no such field: was '
                f'it written for another variant (a schedule, a generator)?') if m else ''
        raise SystemExit(f'config error: {e}{hint}') from e
    if show:
        print(OmegaConf.to_yaml(cfg))
        print(flags_equivalent(cfg, experiment))
        return
    cfg.run.note = check_launch_note(cfg, overrides)     # stripped, as recorded
    train.main(experiment, config=cfg, sources=sources)


def check_launch_note(cfg, overrides):
    """``run.note`` is required, non-empty, and given by the launcher (an
    override on this command line), never by a preset or experiment file."""
    from mapping.provenance import NOTE_HELP, check_note
    given = [o for o in overrides if _key(o) == 'run.note']
    note = OmegaConf.select(cfg, 'run.note')
    if note is not None and not given:
        raise SystemExit(f'run.note={note!r} comes from a file (a preset, an experiment '
                         f'file or a saved run), not from this launch. {NOTE_HELP}')
    return check_note(note)


if __name__ == '__main__':
    main()
