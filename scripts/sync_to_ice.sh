#!/usr/bin/env bash
# Push this project folder to ICE scratch. Does not sync tools/ or the umbrella repo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$ROOT" rev-parse HEAD > "${ROOT}/.git_commit"
fi
rsync -a --progress \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'data' \
  --exclude 'checkpoints' \
  --exclude 'logs' \
  --exclude 'logs-ice' \
  --exclude 'outputs' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  "${ROOT}/" \
  nponna3@login-ice.pace.gatech.edu:~/scratch/cv21/p01-efficient-backbone/
