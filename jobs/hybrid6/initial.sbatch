#!/bin/bash
# Approved initial hybrid6 experiment. Isolated deployment; no other jobs changed.
#SBATCH --job-name=map-hybrid6-s
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ax2348/mapping_hybrid6/results/jobs/logs/initial-%j.out
set -euo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_hybrid6
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/scratch/ax2348/mapping_hybrid6/results/jobs/matplotlib
python scripts/check_env.py
echo "code $(git rev-parse HEAD) job $SLURM_JOB_ID"
python -m experiments.hybrid6.gpu_preflight
python run.py +experiment=hybrid6_initial \
    'run.note=Hybrid6 calibrated camera tokens: causal inert s, SE2d SAW T32, seed0, first 100k steps matched to hybrid4 parent 563beb; approved initial Torch run'
