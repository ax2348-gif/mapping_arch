#!/bin/bash
# Long-chain causal+inert on se2d (user, 2026-10-17, option C): does causal+inert
# compose over chains se2d at T=32 barely trains (median query age 5.3)?
# Scaling recipe (se2d_scaling_wsd, 400k WSD), query_mode=inert, 8 queries.
#   0 hybrid4_causal  s  T=64  path=saw   (no loop closures; exact rejection, accept 0.11%)
#   1 hybrid4_causal  xs T=64  path=saw
#   2 hybrid4_causal  s  T=128 path=walk  (closures present; rerun the loop-closure probe)
#   3 vanilla4_causal s  T=128 path=walk  (architecture contrast)
# References: H1f normal-query causal/bidirectional runs at T=64/128.
# Runs from the git clone /scratch/ax2348/mapping_saw.
#SBATCH --job-name=map-longchain
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/ax2348/mapping_saw/results/jobs/logs/longchain-%A_%a.out
set -euo pipefail
MODELS=(hybrid4_causal hybrid4_causal hybrid4_causal vanilla4_causal)
SIZES=(s xs s s)
TS=(64 64 128 128)
PATHS=(saw saw walk walk)
i=$SLURM_ARRAY_TASK_ID
M=${MODELS[$i]}; S=${SIZES[$i]}; T=${TS[$i]}; P=${PATHS[$i]}
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_saw
python scripts/check_env.py || exit 1    # dependency preflight (requirements.txt)
echo "code $(git rev-parse HEAD) job $SLURM_JOB_ID task $i: $M $S T=$T path=$P"
python run.py +experiment=se2d_scaling_wsd model=se2d/$M model.size=$S model.query_mode=inert \
    task.episode_steps=$T task.generator.path=$P task.generator.workers=14 \
    run.save_every=25000 run.name=inert_seed0_400k_qt_wsd_t${T}_${P} \
    "run.note='Long-chain causal+inert: $M $S, T=$T, path=$P, 400k WSD (option C)'"
