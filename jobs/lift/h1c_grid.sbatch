#!/bin/bash
# H1c grid (HYPOTHESIS.md 2026-10-16 amendment): causal vs bidirectional hybrid4 /
# vanilla4 on se2d, 400k WSD scaling recipe, seeds 0-2, sizes s m l, plus the
# T=64 s causal pair. Via Hydra from /scratch/ax2348/mapping_hydra (exp/causal).
# Already running elsewhere: vanilla4_causal s seed0 (local), hybrid4_causal s
# seed0 and vanilla4_causal l seed0 (array 20070043). Bidirectional seed-0 (all
# sizes), seed-1 s and T=64 s references exist in /scratch/ax2348/mapping/results/se2/.
#SBATCH --job-name=map-h1c
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=h200_public,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-22%10
#SBATCH --output=/scratch/ax2348/mapping_hydra/jobs/logs/h1c-%A_%a.out
set -uo pipefail
source /scratch/ax2348/venvs/mapping/bin/activate
cd /scratch/ax2348/mapping_hydra
python -c "import hydra, omegaconf, fla" || { echo "missing dependency"; exit 1; }
line=$(grep -v '^#' jobs/h1c_tasks.txt | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "code $(cat LIFT_COMMIT) task $SLURM_ARRAY_TASK_ID: $line"
run() {  # model size seed T workers
  local name=seed${3}_400k_qt_wsd; [ "$4" = 64 ] && name=${name}_t64
  python run.py +experiment=se2d_scaling_wsd model=se2d/$1 model.size=$2 run.seed=$3 \
      run.name=$name task.episode_steps=$4 task.generator.workers=$5
}
IFS='|' read -ra parts <<< "$line"
if [ ${#parts[@]} -eq 1 ]; then
  run ${parts[0]} 14
else
  pids=()
  for p in "${parts[@]}"; do run $p 7 & pids+=($!); done
  rc=0; for pid in "${pids[@]}"; do wait $pid || rc=1; done; exit $rc
fi
