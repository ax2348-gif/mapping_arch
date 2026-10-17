#!/bin/bash
# SAW matched pairs on se2d (user, 2026-10-17): does training on walks without
# any revisit (task.generator.path=saw: the walk conditioned on no revisit, exact
# rejection, accept 4.8%) change what the reference runs learned on the walk?
# Each run's args record equals its walk reference's except path (checked with
# mapping.config legacy_args against each reference's final.json):
#   0 hybrid4  s  mixed, 400k WSD, seed 0  vs results/se2/hybrid4/s/seed0_400k_qt_wsd_small
#     packed with
#     hybrid4  xs mixed, 100k cosine, seed 1 vs results/se2/hybrid4/xs/seed1_100k_qt
#     (xs reference ran with workers 4; here 6, the batch sequence does not depend on it)
#   1 vanilla4 l  mixed, 400k WSD, seed 0  vs results/se2/vanilla4/l/seed0_400k_qt_wsd
#     (architecture contrast)
# Local seed 0 xs result (2026-10-17, erid-2 run 2f3333): saw 0.0211 vs walk 0.0180;
# the walk-trained model scores 0.0212 on saw episodes (experiments/se2d_saw/).
# Runs from the git clone /scratch/ax2348/mapping_saw: bash jobs/submit.sh jobs/saw/matched_pairs.sbatch
#SBATCH --job-name=map-sawpairs
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/ax2348/mapping_saw/results/jobs/logs/sawpairs-%A_%a.out
set -euo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_saw
python scripts/check_env.py || exit 1    # dependency preflight (requirements.txt)
i=$SLURM_ARRAY_TASK_ID
echo "code $(git rev-parse HEAD) job $SLURM_JOB_ID task $i"
if [ "$i" = 0 ]; then
    python run.py +experiment=se2d_scaling_wsd model=se2d/hybrid4 model.size=s \
        task.generator.path=saw run.name=seed0_400k_qt_wsd_saw \
        "run.note='SAW matched pair: hybrid4 s mixed 400k WSD seed0 vs walk seed0_400k_qt_wsd_small'" &
    S=$!
    python run.py model=se2d/hybrid4 model.size=xs run.seed=1 run.steps=100000 \
        eval.every=5000 eval.log_every=1000 task.query_times=uniform \
        task.generator.path=saw task.generator.workers=6 run.name=seed1_100k_qt_saw \
        "run.note='SAW matched pair: hybrid4 xs 100k cosine seed1 vs walk seed1_100k_qt (packed with s)'" &
    X=$!
    rc=0; wait $X || rc=$?; wait $S || rc=$?; exit $rc
else
    python run.py +experiment=se2d_scaling_wsd model=se2d/vanilla4 model.size=l \
        task.generator.path=saw run.name=seed0_400k_qt_wsd_saw \
        "run.note='SAW matched pair: vanilla4 l mixed 400k WSD seed0 vs walk seed0_400k_qt_wsd (architecture contrast)'"
fi
