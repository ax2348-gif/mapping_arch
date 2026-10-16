#!/bin/bash
# Run once on the Torch cluster as a CPU job:
#   sbatch --account=torch_pr_1024_tandon_advanced --cpus-per-task=4 --mem=16G --time=00:40:00 scripts/pyhdr.sh
# Installs Python 3.12 headers from conda-forge (the default Anaconda channels
# need their Terms of Service accepted) so Triton can compile its CUDA helper,
# and points C_INCLUDE_PATH at them from the venv activate script.
set -euo pipefail
module load anaconda3/2026.07
conda create -y -q -p /scratch/ax2348/envs/py312hdr --override-channels -c conda-forge python=3.12
test -f /scratch/ax2348/envs/py312hdr/include/python3.12/Python.h
echo "export C_INCLUDE_PATH=/scratch/ax2348/envs/py312hdr/include/python3.12\${C_INCLUDE_PATH:+:\$C_INCLUDE_PATH}" >> /scratch/ax2348/venvs/mapping/bin/activate
source /scratch/ax2348/venvs/mapping/bin/activate
echo "#include <Python.h>
int main(void){return 0;}" > /tmp/pyh_test_$$.c
gcc -fsyntax-only /tmp/pyh_test_$$.c && echo "Python.h compiles OK"
