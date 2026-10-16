#!/usr/bin/env bash
# Sync this project with the Torch cluster scratch space.
#   scripts/sync.sh push    code + the MNIST bank up
#   scripts/sync.sh pull    results and job logs down
# MAPPING_REMOTE overrides the remote tree (tests point it at a local dir).
set -euo pipefail
REMOTE=${MAPPING_REMOTE:-torch:/scratch/ax2348/mapping}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# Run directories are results/<task>/<model>/<size>/<run>/ (run.dir). Trees
# from before 2026-10-16 keep z2 runs at results/<model>/...: everything at
# the top of the remote results/ that is not a task directory (or jobs/) is
# such a z2 entry and lands under the local results/z2/, so a pull never
# re-creates the old layout here (docs/config.md, "Run directories").
TOP=(se2 se2d se2c z2 jobs)
case "${1:-}" in
  push)
    rsync -az --info=stats1 \
      --exclude results --exclude jobs/logs --exclude '__pycache__' \
      --exclude '*.pt' --exclude '.git' \
      "$HERE/" "$REMOTE/"
    ;;
  pull)
    keep=(); skip=()
    for t in "${TOP[@]}"; do keep+=(--include "/$t/***"); skip+=(--exclude "/$t"); done
    mkdir -p "$HERE/results/z2"
    rsync -az --info=stats1 "${keep[@]}" --exclude '*' "$REMOTE/results/" "$HERE/results/"
    rsync -az --info=stats1 "${skip[@]}" "$REMOTE/results/" "$HERE/results/z2/"
    rsync -az --info=stats1 "$REMOTE/jobs/logs/" "$HERE/jobs/logs/" || true
    ;;
  *)
    echo "usage: $0 push|pull" >&2; exit 2
    ;;
esac
