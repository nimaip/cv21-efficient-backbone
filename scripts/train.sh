#!/usr/bin/env bash
# Local training entry. Uses the project venv when it exists.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="${PYTHON:-python}"
fi
exec "$PY" -m efficient_backbone.train "$@"
