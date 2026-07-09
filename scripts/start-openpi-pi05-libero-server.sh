#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/data/L202500340/New-IL/third_party/openpi}"
OPENPI_CONDA_ENV="${OPENPI_CONDA_ENV:-openpi}"
PORT="${PORT:-8000}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data/L202500340/data/openpi}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
if ! command -v conda >/dev/null 2>&1; then
  for _conda_bin in /root/miniconda3/bin "$HOME/miniconda3/bin" /opt/conda/bin; do
    if [[ -x "${_conda_bin}/conda" ]]; then
      export PATH="${_conda_bin}:${PATH}"
      break
    fi
  done
fi

cd "$OPENPI_ROOT"
export OPENPI_DATA_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION
exec conda run --no-capture-output -n "$OPENPI_CONDA_ENV" \
  python scripts/serve_policy.py --env LIBERO --port "$PORT" "$@"
