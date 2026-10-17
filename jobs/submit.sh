#!/bin/bash
# Submit a Torch job only after the dependency preflight passes, on the login
# node, in the job's own venv: a missing package fails here in seconds instead
# of after the queue wait (job 20101203 died on scipy 32 s after starting).
#   jobs/submit.sh jobs/lift/<job>.sbatch [extra sbatch args]
# Run from the job's git clone on Torch. The sbatch must also run the preflight
# itself (`python scripts/check_env.py || exit 1`), which this script checks.
set -euo pipefail
cd "$(dirname "$0")/.."
job=$1; shift
[ -f "$job" ] || { echo "submit: no such file $job"; exit 1; }
grep -q 'scripts/check_env.py' "$job" \
    || { echo "submit: $job does not run scripts/check_env.py before training"; exit 1; }
venv=$(grep -oP 'source \K\S+(?=/bin/activate)' "$job" | head -1)
[ -n "$venv" ] || { echo "submit: $job does not activate a venv"; exit 1; }
source "$venv/bin/activate"
python scripts/check_env.py || { echo "submit: preflight failed in $venv; nothing submitted"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] \
    || { echo "submit: tracked files modified in $(pwd); commit or sync first"; exit 1; }
echo "submit: code $(git rev-parse --short HEAD)"
sbatch "$job" "$@"
