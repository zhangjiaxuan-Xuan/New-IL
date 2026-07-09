#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
LIBERO_ROOT="${LIBERO_ROOT:-/data/L202500340/New-IL/third_party/LIBERO}"
NEWIL_CONDA_ENV="${NEWIL_CONDA_ENV:-newil}"

exec conda run --no-capture-output -n "$NEWIL_CONDA_ENV" \
  python -m new_il.libero.eval_server \
  --host "$HOST" \
  --port "$PORT" \
  --libero-root "$LIBERO_ROOT" \
  "$@"
