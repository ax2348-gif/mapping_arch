#!/bin/bash
# vanilla4c s on the DEFAULT continuous se2c task, 100k cosine: matched to
# hybrid4c s dec_100k_qt (worker 20041129). Runs from the separate lift tree.
#SBATCH --job-name=map-se2c-dec-vanilla4c-s-100k
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ax2348/mapping_lift/jobs/logs/se2c-dec-vanilla4c-s-100k-%j.out
set -euo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_lift
echo "code $(cat LIFT_COMMIT 2>/dev/null)"
python train.py --task se2c --model vanilla4c --size s --steps 100000 --query-times uniform \
    --eval-every 2500 --log-every 500 --save-every 5000 --run dec_100k_qt
