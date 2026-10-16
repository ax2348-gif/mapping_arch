#!/usr/bin/env bash
# One-time environment on the Torch cluster. Run on a login node:
#   ssh torch 'bash -s' < scripts/remote_setup.sh
set -euo pipefail
VENV=/scratch/ax2348/venvs/mapping
mkdir -p /scratch/ax2348/mapping/jobs/logs
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch numpy
# the config tree's dependencies, pinned (requirements.txt; every job checks them)
if [ -f /scratch/ax2348/mapping/requirements.txt ]; then
    pip install -r /scratch/ax2348/mapping/requirements.txt
else
    echo "after scripts/sync.sh push: pip install -r /scratch/ax2348/mapping/requirements.txt"
fi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
