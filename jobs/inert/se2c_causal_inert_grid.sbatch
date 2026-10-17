#!/bin/bash
# se2c causal+inert (user, 2026-10-17): hybrid4c_causal vs vanilla4c_causal with
# model.query_mode=inert on the default continuous task, the se2c ladder recipe
# (WSD, checkpoints every 25k, eval 5k, log 1k, uniform query times), matching
# the mixed bidirectional ladder (results/se2c/{hybrid4c,vanilla4c}/{m,l}/dec_*_wsd).
#   0 hybrid4c_causal m 400k   1 vanilla4c_causal m 400k
#   2 hybrid4c_causal l 200k   3 vanilla4c_causal l 200k
# Runs from the git clone /scratch/ax2348/mapping_se2c_ci.
#SBATCH --job-name=map-se2c-cinert
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/ax2348/mapping_se2c_ci/results/jobs/logs/se2c-cinert-%A_%a.out
set -euo pipefail
MODELS=(hybrid4c_causal vanilla4c_causal hybrid4c_causal vanilla4c_causal)
SIZES=(m m l l)
STEPS=(400000 400000 200000 200000)
i=$SLURM_ARRAY_TASK_ID
M=${MODELS[$i]}; S=${SIZES[$i]}; N=${STEPS[$i]}
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_se2c_ci
python scripts/check_env.py || exit 1    # dependency preflight (requirements.txt)
echo "code $(git rev-parse HEAD) job $SLURM_JOB_ID task $i: $M $S $N"
python run.py +experiment=se2c_causal model=se2c/$M model.size=$S model.query_mode=inert schedule=wsd \
    run.steps=$N run.save_every=25000 eval.every=5000 eval.log_every=1000 \
    run.name=inert_dec_$((N / 1000))k_wsd \
    "run.note='se2c causal+inert: $M $S, $((N / 1000))k WSD, ladder recipe'"
