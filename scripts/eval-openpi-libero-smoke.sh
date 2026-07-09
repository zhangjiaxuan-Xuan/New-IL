#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LIBERO_ROOT="${LIBERO_ROOT:-/data/L202500340/New-IL/third_party/LIBERO}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/openpi_libero_smoke}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL MUJOCO_EGL_DEVICE_ID PYOPENGL_PLATFORM

exec conda run --no-capture-output -n newil \
  python -c "from new_il.libero.rollout import openpi_rollout_main; openpi_rollout_main()" \
  --host "$HOST" \
  --port "$PORT" \
  --libero-root "$LIBERO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --task-suite-name "${TASK_SUITE_NAME:-libero_spatial}" \
  --task-id "${TASK_ID:-0}" \
  --episode-idx "${EPISODE_IDX:-0}" \
  "$@"
