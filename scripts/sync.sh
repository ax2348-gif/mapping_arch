#!/usr/bin/env bash
# Sync this project with the Torch cluster scratch space.
#   scripts/sync.sh push    code + the MNIST bank up
#   scripts/sync.sh pull    results and job logs down
set -euo pipefail
REMOTE=torch:/scratch/ax2348/mapping
HERE="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  push)
    rsync -az --info=stats1 \
      --exclude results --exclude jobs/logs --exclude '__pycache__' \
      --exclude '*.pt' --exclude '.git' \
      "$HERE/" "$REMOTE/"
    ;;
  pull)
    rsync -az --info=stats1 "$REMOTE/results/" "$HERE/results/"
    rsync -az --info=stats1 "$REMOTE/jobs/logs/" "$HERE/jobs/logs/" || true
    ;;
  *)
    echo "usage: $0 push|pull" >&2; exit 2
    ;;
esac
