#!/usr/bin/env bash
# Optional site-neutral launcher; supply allocation settings to sbatch.
set -euo pipefail
config=${1:?Usage: sbatch [resources] run/slurm.sh CONFIG [train|train_with_downstream]}
mode=${2:-train_with_downstream}
case "$mode" in
  train|train_with_downstream) ;;
  *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
esac
cd "${SLURM_SUBMIT_DIR:-$PWD}"
repro_python=${REPRO_PYTHON:-.venv/bin/python}
exec "$repro_python" main.py "$mode" --config "$config"
