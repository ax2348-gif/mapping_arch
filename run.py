"""Train one run from the Hydra config tree (``conf/``, docs/config.md).

    python run.py model=se2d/hybrid4 model.size=s run.steps=500
    python run.py +experiment=se2c_bridge model.size=m
    python run.py --exp path/to/my_experiment.yaml run.seed=1
    python run.py --from results/se2c/hybrid4c/m/<run>/config.yaml run.name=rerun
    python run.py ... --print            # the resolved config and train.py flags; no run

Arguments are Hydra overrides (``key=value``, ``group=option``,
``+experiment=name``). ``--exp FILE`` composes an experiment file from
anywhere exactly like a preset in ``conf/experiment/``. ``--from FILE``
reruns a run's saved ``config.yaml`` (fully resolved, so only value
overrides apply, not group choices).

The tree is composed with Hydra's compose API rather than ``@hydra.main``,
whose command-line parser fails on Python 3.14 ("badly formed help
string") and which would change the working directory and logging.
"""
import os
import shutil
import sys
import tempfile

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import mapping.config  # noqa: F401  (registers the model_lr resolver)

CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf')


def load(overrides=(), exp=None):
    """The composed config: ``conf/config.yaml`` with ``overrides``, and
    the experiment file ``exp`` (any path) as ``+experiment=<its name>``."""
    overrides = list(overrides)
    with tempfile.TemporaryDirectory() as extra:
        if exp is not None:
            os.makedirs(os.path.join(extra, 'experiment'))
            name = os.path.splitext(os.path.basename(exp))[0]
            shutil.copy(exp, os.path.join(extra, 'experiment', name + '.yaml'))
            overrides = [f'hydra.searchpath=[file://{extra}]',
                         f'+experiment={name}'] + overrides
        with initialize_config_dir(version_base='1.3', config_dir=CONF):
            return compose('config', overrides=overrides)


def resolved(cfg):
    """The config with every interpolation resolved, as runs save it."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def build(cfg):
    """The ``mapping.config.Experiment`` a config describes."""
    return instantiate(cfg, _convert_='all')


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
        cfg = OmegaConf.load(frm)
        cfg.merge_with(OmegaConf.from_dotlist(overrides))
    else:
        cfg = resolved(load(overrides, exp))
    experiment = build(cfg)
    args = experiment.legacy_args()
    if show:
        print(OmegaConf.to_yaml(cfg))
        print('train.py ' + ' '.join(train.flags(args)))
        return
    train.main(args, config=cfg)


if __name__ == '__main__':
    main()
