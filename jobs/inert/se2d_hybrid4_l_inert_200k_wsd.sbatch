#!/bin/bash
# Inert-query capacity check (user, 2026-10-17): hybrid4 l on se2d with
# model.query_mode=inert, the scaling recipe (se2d_scaling_wsd) cut to 200k.
# xs inert stalled at 0.080 (mixed 0.018); this rules out a capacity limit.
# Reference: results/se2/hybrid4/l/seed0_400k_qt_wsd (mixed), compared at
# matched stable-phase steps. Runs from the git clone /scratch/ax2348/mapping_inert.
#SBATCH --job-name=map-inert-l
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ax2348/mapping_inert/results/jobs/logs/inert-l-%j.out
set -euo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_inert
python scripts/check_env.py || exit 1    # dependency preflight (requirements.txt)
echo "code $(git rev-parse HEAD) job $SLURM_JOB_ID"
python run.py +experiment=se2d_scaling_wsd model.size=l model.query_mode=inert \
    run.steps=200000 run.save_every=25000 run.name=inert_seed0_200k_qt_wsd task.generator.workers=14 \
    "run.note='Inert-query capacity check: hybrid4 l se2d inert, 200k WSD (xs inert stalled at 0.080 vs mixed 0.018)'"
