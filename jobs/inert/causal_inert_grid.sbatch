#!/bin/bash
# Causal+inert architecture comparison (user, 2026-10-17): vanilla4_causal vs
# hybrid4_causal with model.query_mode=inert on se2d, scaling recipe
# (se2d_scaling_wsd). Matched to hybrid4_causal s inert (local, 400k) and
# hybrid4_causal l inert (Torch 20193008, 200k). Normal-query causal vanilla4
# failed at s and solved at l, so m brackets its threshold.
#   0 vanilla4_causal s 400k   1 vanilla4_causal m 400k
#   2 vanilla4_causal l 200k   3 hybrid4_causal  m 400k
# Runs from the git clone /scratch/ax2348/mapping_inert.
#SBATCH --job-name=map-cinert-grid
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/ax2348/mapping_inert/results/jobs/logs/cinert-grid-%A_%a.out
set -euo pipefail
MODELS=(vanilla4_causal vanilla4_causal vanilla4_causal hybrid4_causal)
SIZES=(s m l m)
STEPS=(400000 400000 200000 400000)
i=$SLURM_ARRAY_TASK_ID
M=${MODELS[$i]}; S=${SIZES[$i]}; N=${STEPS[$i]}
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_inert
python scripts/check_env.py || exit 1    # dependency preflight (requirements.txt)
echo "code $(git rev-parse HEAD) job $SLURM_JOB_ID task $i: $M $S $N"
python run.py +experiment=se2d_scaling_wsd model=se2d/$M model.size=$S model.query_mode=inert \
    run.steps=$N run.save_every=25000 run.name=inert_seed0_$((N / 1000))k_qt_wsd task.generator.workers=14 \
    "run.note='Causal+inert architecture comparison: $M $S se2d inert, $((N / 1000))k WSD'"
