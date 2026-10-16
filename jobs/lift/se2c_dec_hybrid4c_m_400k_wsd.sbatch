#!/bin/bash
# se2c data scaling: hybrid4c m on the DEFAULT continuous task (decoupled
# persistent walk, continuous turns), 4x the data of the 100k runs, under
# warmup-stable-decay with checkpoints every 25k (cooldown branches give the
# shorter budgets). Runs from the separate lift tree.
#SBATCH --job-name=map-se2c-dec-hybrid4c-m-400k
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/ax2348/mapping_lift/jobs/logs/se2c-dec-m-400k-%j.out
set -euo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_lift
echo "code $(cat LIFT_COMMIT 2>/dev/null)"
python train.py --task se2c --model hybrid4c --size m --steps 400000 --query-times uniform \
    --schedule wsd --save-every 25000 --eval-every 5000 --log-every 1000 --run dec_400k_wsd
