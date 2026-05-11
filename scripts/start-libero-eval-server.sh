#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/x/Xcode/Mem/third_party/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/x/Xcode/Mem/.venvs/libero/bin/python}"

exec "$LIBERO_PYTHON" -m new_il.libero.eval_server \
  --host "$HOST" \
  --port "$PORT" \
  --libero-root "$LIBERO_ROOT" \
  "$@"
