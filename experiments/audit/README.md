# Provenance audit, 2026-10-16 (Phase 1)

A read-only inventory of every training run that exists, and how far each one can be traced
back to code and config. Nothing was launched, cancelled or modified; Torch was read over ssh.
The design that follows from this audit is `docs/provenance_design.md` (Phase 2).

Files:
- `runs.csv` holds every run directory found, one row per run (215 rows), with the brief's per-run fields.
- `../registry.yaml` is the curated list of MAJOR experiments (13), plus the pending and borderline items.

## Method

1. **Run directories.** A run directory is any directory holding `log.jsonl`, `final.json` or
   `config.yaml`. I walked these places:
   - local: `~/code/mapping/{results,shelved,experiments}`, `~/code/mapping-hydra/results`,
     `~/code/mapping-hydra-seams/results`, `~/code/mapping-odq`, `~/code/mapping-ocean`, and the
     session scratchpad;
   - Torch, read-only: `/scratch/ax2348/{mapping,mapping_lift,mapping_hydra,mapping_odq}`.

   The blind probe's own harness (`experiments/lift_blind_probe/runs/`) uses other file names, so
   I added its rows by hand.
2. **Identity and copies.** A run's id is `<host>:<task>/<model>/<size>/<run>`, with `se2` written as
   `se2d`.
   - **Mirror.** A local directory at the same relative path as a Torch run is a mirror (from
     `scripts/sync.sh pull`) when lines 2-4 of the two `log.jsonl` files hash equal. These lines
     carry wall-clock times, which differ between any two runs. A mirror is folded into the Torch
     row (column `local_mirror`).
   - **Collision.** When the lines differ, the two are different runs (a collision) and both are
     kept.
3. **Jobs.**
   - **Job id.** Every Torch job log prints `<task> <model>/<size> -> results/...`. These lines map
     each run to its job id and log file. The job script comes from the log name and the
     `--output` pattern.
   - **GPU time.** It comes from `sacct -u ax2348 -S 2026-10-10` (ElapsedRaw). Where two runs share
     one GPU (H1c/H1d packing), the job's time is split evenly. Tasks inside the lift worker jobs use
     the run's own `elapsed_s`.
   - **Local runs.** They have no job record. Their GPU time is the run's `elapsed_s` (wall clock on
     the local RTX PRO 4000).
4. **Commits.** No run directory records a commit: neither `train.py` nor `run.py` writes one. The
   commit column takes one of three forms:
   - **Exact sha.** Only where the job log echoed the tree's `LIFT_COMMIT` file at launch (the
     mapping_lift and mapping_hydra jobs). These trees are rsync exports, so even an exact sha says
     nothing about uncommitted edits.
   - **`inferred:<sha>`.** The last commit on the branch the run was launched from, at or before the
     submit time (Torch, from sacct) or the estimated start (local: last write minus `elapsed_s`).
     The branch is master (`ref/pre-hydra` first-parent history) for `~/code/mapping` and the Torch
     `mapping` tree. For `~/code/mapping-hydra` it is exp/causal or exp/time-recurrence.
   - **`unknown`.** The run predates the first commit.

   As a cross-check, every Torch tree's files were hashed as git blobs and matched against every
   commit:
   - `mapping_hydra` is byte-identical to 50341e4 and `mapping_odq` to a353d1e.
   - `mapping_lift` matches 038f19c, its current `LIFT_COMMIT`, apart from five blind-probe files.
   - `mapping` matches a412327, 09215ba and cc7ae4b, apart from `peek*.py`. It is the state of the
     LAST push (04:05), so it says nothing about the code that ran the 2026-10-15 jobs.
5. **Config source.** There are three kinds:
   - `hydra` means `config.yaml` plus `overrides.yaml` (rerunnable with `run.py --from`);
   - `legacy` means the `args` dict in `final.json` or `log.jsonl`, reconstructable as `train.py`
     flags and through the frozen translator;
   - `none` means neither exists.
6. **Major.** Runs were grouped into experiments and scored with the brief's heuristics:
   documentation (STATUS, journal, published report), hypothesis (HYPOTHESIS.md prediction,
   amendment or evidence), downstream (parent checkpoint, reference cell, figure source) and GPU-h.
   - **MAJOR** if hypothesis or downstream, or documented and more than about 1 GPU-h.
   - The report citations come from reading the six published pages and the journal (read-only).
7. **Smoke runs.** A run that trained 3,000 steps or fewer, or a `probe*` or equivalence run, is
   classed as `smoke/test`. These rows are kept in `runs.csv` but excluded from the statistics.

The audit scripts are not committed, since this phase adds no code. They are in the session
scratchpad (`inv.py`, `build.py`, `groups.py`, `registry.py`), and `runs.csv` can be regenerated
from them. Whether to keep them is open question 9 of the design.

## Counts

| | runs | GPU-h |
| --- | --- | --- |
| rows in runs.csv | 215 | 125 |
| smoke or test runs (scratchpad and probes) | 56 | <0.5 |
| **real runs** | **159** (115 Torch, 44 local) | **125** (115 Torch, 10 local) |
| in the 13 MAJOR experiments | 147 | 123 |
| in the 5 borderline groups | 12 | 2.4 |
| status: completed / running / stopped | 138 / 10 / 11 | |

- **Cross-check.** sacct reports 116.5 GPU-h for `map-*` GPU jobs, against 115.3 in the
  inventory's Torch rows.
- **Pending.** The registry lists 20072852_[3-22], 20076519_[1-5] and 20084102_[0-2] under
  `pending`. They have no run directory yet.

## Gap statistics (159 real runs)

| gap | runs | share |
| --- | --- | --- |
| no recorded commit (inferred from time) | 136 | 86% |
| no commit even inferable (run predates the first commit) | 4 | 3% |
| exact commit recorded (LIFT_COMMIT echoed in the job log) | 19 | 12% |
| tree clean at launch, verifiable | **0** | **0%** |
| config only as legacy train.py args | 136 | 86% |
| config as Hydra config.yaml + overrides.yaml | 17 | 11% |
| config missing (cancelled legacy runs before any final.json; blind-probe cdiag) | 3 | 2% |
| local run with no job or launch record (command lives in agent transcripts or the scratchpad) | 44 | 28% |
| launched from code that, at launch, was only on a branch (exp/causal, exp/time-recurrence; now merged) | 7 | 4% |
| training entry point other than train.py / run.py (blind probe) | 4 | 3% |
| stopped without final.json or any checkpoint | 4 | 3% |
| two different runs at one results path | 1 pair | |
| local mirror is stale (looks final, is not) | 9 of 94 mirrors | |

## Borderline, for the user's review

**Decided (the user, 2026-10-16):** the pre-freeze ablations and the
time-recurrence pair are major (moved into the registry's `experiments:`),
the original `hybrid` sweep is major, the blind probe is not. The proving run
and the VO worker had no call and stay borderline. The major runs (153) are
in the ledger as retroactive records (`tools/import_retro.py`;
docs/provenance_design.md §15, phase 3). Local z2 paths below moved to
`results/z2/` (`z2_results_migration.tsv`).

These are documented but cheap, or similar. None is counted as major; the registry's
`borderline:` block lists each group's runs.

1. **se2d-prefreeze-ablations** (1.1 GPU-h): conv muP init, conv-width race (paused at 78k),
   hybrid4.5b and hybrid45b_last at s.
   - Only a STATUS line documents them. The promised `docs/lift/prefreeze_ablations.md` does not
     exist.
   - Either write them up (then major) or mark them abandoned.
2. **se2d-time-recurrence** (0.4 GPU-h): hybrid4_time and _full at xs. Documented in STATUS,
   `docs/time_recurrence.md` and the se2c report, but there is no registered prediction.
   - It reads like a finding (the time scan learns the rotation), so I lean major.
3. **lift-blind-probe** (0.5 GPU-h, shelved). It also has its own training entry point outside the
   seal (`experiments/lift_blind_probe/train.py`).
4. **se2c-proving-run** (hybrid4c xs 25k, 0.15 GPU-h). It is cited in STATUS and the journal, and
   superseded by the bridge.
5. **se2c-vo-worker** (vo_120k_qt, died at 13k). It is documented as a failure and is replaced by the
   exp/odq baseline.

Also for review:
- **z2-day1-local** was scored major only because the frozen evidence cites it (0.0150 at 20k). It
  is 0.4 GPU-h in all.
- **se2d-hybrid-original-sweep** (25.8 GPU-h) is major on documentation plus GPU time. Its only
  hypothesis link is one sentence of an amendment.

## Reproducibility gaps by severity

### Critical (results whose code cannot be pinned, or data at risk)

1. **No run records its code.**
   - `train.py` and `run.py` write no commit, no dirty flag and no environment. 136 of 159 runs carry
     only an inferred commit.
   - The 19 "exact" ones echo a hand-maintained `LIFT_COMMIT` file. Nothing ties it to the bytes that
     ran.
2. **The Torch trees are rsync exports with no `.git`.** `scripts/sync.sh push` copies the working
   tree, dirty edits included.
   - `mapping` has no commit file at all. Every push overwrote the code of earlier jobs: the tree now
     matches a412327 (04:05 on 10-16), not the code that ran the 10-15 sweeps (inferred 7c1f827,
     77ce224, 6630e45).
   - `LIFT_COMMIT` in mapping_lift is overwritten on each push. The ladder jobs echo 8837c28, which is
     a STATUS-only commit at 06:39. The file now says 038f19c.
3. **Path collision, with data loss pending.**
   - `results/se2c/hybrid4c/m/bridge_n4dec_100k_qt` holds two different runs: local, 23.5k steps, and
     Torch 20039511, 68k.
   - The next `scripts/sync.sh pull` (rsync) will overwrite the local run with the Torch one. It
     should be renamed or copied aside before the next pull. I did not touch it.
4. **Published figures were built from ephemeral copies.** All three snapshots live in the session
   scratchpad, which is not durable:
   - the Z² report from `scratchpad/z2report/build.py`;
   - the se2c report from a pull into `scratchpad/writeup/torch` (12:33);
   - the se2d scaling analysis in `scratchpad/scaling/`.

   No report records the exact run set and step it read, beyond a "data pulled" time.

### High

5. **Code exists only on a branch.**
   - The odq array 20084102 runs a353d1e. That commit is on exp/odq in `~/code/mapping-hydra` and is
     absent from `~/code/mapping`.
   - If that clone is lost, the code of those runs is lost with it.
6. **Legacy configs are flags.** 136 runs are reconstructable only as `train.py` flags.
   - Rerunning them goes through the frozen translator at an inferred commit, so a rerun is not an
     exact reproduction.
   - Two cancelled legacy runs (hybrid s 400k cosine and WSD) have no args at all. Their sbatch files
     (`jobs/legacy/`, gitignored) are the only record.
7. **Local runs have no launch record (44).** Commands live in agent transcripts and scratchpad logs
   (`scratchpad/{timescan,prefreeze,lift_env}/*.log`). `jobs/legacy/` and `jobs/logs/` are
   gitignored, so the sbatch scripts of all pre-Hydra Torch runs are untracked local copies.
8. **Stale mirrors.** 9 local mirrors stopped short of their Torch run. For example, `results/se2/`
   hybrid4 xl locally ends at 182k, while on Torch it reached 400k. Anything reading `results/`
   locally (summarize.py, reports) can silently use partial curves.

### Medium

9. **A second training entry point.** `experiments/lift_blind_probe/train.py` trains outside
   `run.py`. The seal-stage-1 scan should either cover `experiments/` or exempt it explicitly.
10. **Documentation error.** STATUS says "hybrid4 xs seed 1, 20028847_0, was killed at 360k". The
    job log says 20028847_0 is hybrid4 **s** seed 1 (`results/se2/hybrid4/s/seed1_400k_qt_wsd_small`,
    stopped at 366k). I recorded this here and did not edit STATUS.
11. **GPU-time attribution is approximate.** Packed jobs are split 50/50, and the lift worker held
    the GPU idle between tasks, which no run is charged for. Local hours are wall clock.
12. **Four runs predate the first commit** (`*/xs/seed0_8k`, 10-15 before 17:06). Their code is
    unrecoverable.
13. **This is a snapshot.** 10 runs were running and 28 array tasks pending at 16:05 on 10-16. Their final
    status, metrics and GPU time will differ.
