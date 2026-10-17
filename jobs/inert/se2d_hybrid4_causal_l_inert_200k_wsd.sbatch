#!/bin/bash
# Causal+inert verification (user, 2026-10-17): hybrid4_causal l on se2d with
# model.query_mode=inert, the scaling recipe (se2d_scaling_wsd) cut to 200k.
# Pairs with the inert l job (20190702); the causal mixed l reference is from the H1f grid.
# References: the bidirectional mixed l (results/se2/hybrid4/l/seed0_400k_qt_wsd)
# and the inert l job, compared at matched stable-phase steps. Runs from the git clone /scratch/ax2348/mapping_inert.
#SBATCH --job-name=map-cinert-l
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ax2348/mapping_inert/results/jobs/logs/cinert-l-%j.out
set -euo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_inert
python scripts/check_env.py || exit 1    # dependency preflight (requirements.txt)
echo "code $(git rev-parse HEAD) job $SLURM_JOB_ID"
python run.py +experiment=se2d_scaling_wsd model=se2d/hybrid4_causal model.size=l model.query_mode=inert \
    run.steps=200000 run.save_every=25000 run.name=inert_seed0_200k_qt_wsd task.generator.workers=14 \
    "run.note='Causal+inert verification: hybrid4_causal l se2d inert, 200k WSD (bidirectional inert s 0.0024 vs mixed 0.0075 at 100k)'"
