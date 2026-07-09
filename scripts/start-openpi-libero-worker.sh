#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="${QUEUE_DIR:-runs/openpi_libero_queue}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/openpi_libero_collect}"
WORKER_ID="${WORKER_ID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LIBERO_ROOT="${LIBERO_ROOT:-/data/L202500340/New-IL/third_party/LIBERO}"
NEWIL_CONDA_ENV="${NEWIL_CONDA_ENV:-newil}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if ! command -v conda >/dev/null 2>&1; then
  for _conda_bin in /root/miniconda3/bin "$HOME/miniconda3/bin" /opt/conda/bin; do
    if [[ -x "${_conda_bin}/conda" ]]; then
      export PATH="${_conda_bin}:${PATH}"
      break
    fi
  done
fi
export MUJOCO_GL MUJOCO_EGL_DEVICE_ID PYOPENGL_PLATFORM

exec conda run --no-capture-output -n "$NEWIL_CONDA_ENV" \
  python -c "from new_il.libero.openpi_queue import worker_main; worker_main()" \
  --queue-dir "$QUEUE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --worker-id "$WORKER_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --libero-root "$LIBERO_ROOT" \
  "$@"
