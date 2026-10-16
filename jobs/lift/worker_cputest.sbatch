#!/bin/bash
# Lift worker: holds one GPU for the hill climb, with a tmux session INSIDE the
# job (on the compute node). The first task starts in the session the moment
# the GPU is granted. Later runs are typed into the session by hand, and a run
# is stopped with Ctrl-C, never by killing processes:
#   srun --jobid=<id> --overlap tmux send-keys -t lift '<command>' Enter
#   srun --jobid=<id> --overlap tmux send-keys -t lift C-c
#   srun --jobid=<id> --overlap tmux capture-pane -p -t lift | tail
#   sbatch jobs/lift/worker.sbatch "<first command>"
#SBATCH --job-name=map-lift-tmuxtest
#SBATCH --account=torch_pr_1024_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/ax2348/mapping_lift/jobs/logs/worker-%j.out
export PATH=$HOME/.local/bin:$PATH
cd /scratch/ax2348/mapping_lift
tmux new-session -d -s lift -x 250 -y 50
tmux send-keys -t lift "source /scratch/ax2348/venvs/mapping/bin/activate && cd /scratch/ax2348/mapping_lift" Enter
tmux send-keys -t lift "$1 2>&1 | tee -a jobs/logs/worker-$SLURM_JOB_ID-runs.out" Enter
echo "$(date +%T) tmux session 'lift' up; first task: $1 (code $(cat LIFT_COMMIT 2>/dev/null))"
while tmux has-session -t lift 2>/dev/null; do sleep 30; done
