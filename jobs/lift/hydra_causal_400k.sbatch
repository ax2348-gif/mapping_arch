#!/bin/bash
# Causal baselines at fair GPU time on discrete SE(2) (se2d), scaling-study
# recipe (400k WSD, checkpoints every 50k), via Hydra (run.py) from the
# exp/causal export in /scratch/ax2348/mapping_hydra.
#   0: hybrid4_causal s   (bidirectional reference: results/se2/hybrid4/s/seed0_400k_qt_wsd)
#   1: vanilla4_causal l  (bidirectional reference: results/se2/vanilla4/l/seed0_400k_qt_wsd)
# vanilla4_causal s runs locally (the causality agent).
#SBATCH --job-name=map-se2d-causal-400k
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/ax2348/mapping_hydra/jobs/logs/se2d-causal-400k-%A_%a.out
set -euo pipefail
MODELS=(hybrid4_causal vanilla4_causal)
SIZES=(s l)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; S=${SIZES[$SLURM_ARRAY_TASK_ID]}
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_hydra
python -c "import hydra, omegaconf, fla" || { echo "missing dependency (pip install -r requirements.txt)"; exit 1; }
echo "code $(cat LIFT_COMMIT): $M $S"
python run.py +experiment=se2d_scaling_wsd model=se2d/$M model.size=$S
