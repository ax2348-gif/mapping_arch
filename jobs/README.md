# Job launchers

Layout:

    conf/experiment/<name>.yaml   what a run is (Hydra presets: python run.py +experiment=<name>)
    jobs/<campaign>/              how it was launched on the cluster (sbatch, task lists), tracked
    results/jobs/logs/            SLURM output of new jobs (untracked, like every run)

Launchers sit next to the configs, never inside conf/: Hydra composes only
YAML from conf/, and a launcher is a record of a submission, not a setting.

- `jobs/lift/`: the lift campaign on Torch.
  - h1c_* and h1d_*: the H1c/H1d grids (HYPOTHESIS.md), run.py.
  - hydra_causal_400k: run.py.
  - se2c_bridge_*, se2c_dec_*, worker*: argparse `train.py`, from
    /scratch/ax2348/mapping_lift.
  These files are records of what was submitted, from which Torch tree. Their
  `--output` and task-list paths are that tree's (e.g. mapping_hydra keeps
  its task lists at jobs/h1c_tasks.txt), so they are left as they ran.
- Generated launchers (`scaling.py --slurm`, `tools/branch_jobs.py`) write
  their logs to `results/jobs/logs/`, and pass each task an automatic
  `run.note` (`scaling se2d hybrid4 $SIZE $RUN`, `cooldown ...`): every run
  needs one (docs/config.md, "The run note"). The records in `jobs/lift/`
  predate the rule and are left as they were submitted; relaunching one
  from this tree needs a `run.note=...` added.
- `jobs/logs/` (older logs) and `jobs/legacy/` (pre-Hydra launchers of the
  z2 and se2 runs) are local and gitignored.
