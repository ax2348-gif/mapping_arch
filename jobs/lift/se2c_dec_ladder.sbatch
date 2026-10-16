#!/bin/bash
# se2c (default continuous task) ladder extension, WSD with checkpoints every
# 25k so cooldown branches give shorter budgets. Fixed grid (batch OK):
#   0 hybrid4c  l 200k    1 vanilla4c l 200k
#   2 hybrid4c  s 400k    3 vanilla4c s 400k   (s at the m runs' data budget)
# Runs from the separate lift tree (/scratch/ax2348/mapping_lift).
#SBATCH --job-name=map-se2c-dec-ladder
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/ax2348/mapping_lift/jobs/logs/se2c-dec-ladder-%A_%a.out
set -euo pipefail
MODELS=(hybrid4c vanilla4c hybrid4c vanilla4c)
SIZES=(l l s s)
STEPS=(200000 200000 400000 400000)
i=$SLURM_ARRAY_TASK_ID
M=${MODELS[$i]}; S=${SIZES[$i]}; N=${STEPS[$i]}
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_lift
echo "code $(cat LIFT_COMMIT 2>/dev/null): $M $S $N"
python train.py --task se2c --model $M --size $S --steps $N --query-times uniform \
    --schedule wsd --save-every 25000 --eval-every 5000 --log-every 1000 \
    --run dec_$((N / 1000))k_wsd
