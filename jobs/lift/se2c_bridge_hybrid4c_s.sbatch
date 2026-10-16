#!/bin/bash
# se2c bridge: hybrid4c s on the exact discrete walk (decoupled, lattice
# directions, n = 4, 9 px steps) with the new renderer, texture and encoding.
# Runs from the separate lift tree so the scaling study's code stays frozen.
#SBATCH --job-name=map-se2c-bridge-hybrid4c-s
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ax2348/mapping_lift/jobs/logs/se2c-bridge-%j.out
set -euo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_lift
python train.py --task se2c --model hybrid4c --size s --steps 100000 --query-times uniform \
    --eval-every 2500 --log-every 500 --save-every 5000 --motion decoupled --directions lattice --turn-n 4 \
    --v-min 9 --v-max 9 --run bridge_n4dec_100k_qt
