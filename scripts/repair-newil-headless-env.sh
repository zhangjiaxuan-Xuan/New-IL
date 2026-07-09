#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/L202500340/New-IL}"
NEWIL_CONDA_ENV="${NEWIL_CONDA_ENV:-newil}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/newil_uv_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/newil_pip_cache}"

export UV_CACHE_DIR
export PIP_CACHE_DIR

cd "$PROJECT_ROOT"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

echo "[newil-headless] installing EGL/GLVND runtime only"
conda install -y -n "$NEWIL_CONDA_ENV" -c conda-forge libegl=1.7.0 libglvnd=1.7.0

echo "[newil-headless] installing explicit headless Python runtime deps"
conda run --no-capture-output -n "$NEWIL_CONDA_ENV" uv pip install \
  "numpy==1.26.4" \
  "opencv-python-headless==4.10.0.84" \
  "mujoco==3.9.0" \
  "pyopengl==3.1.10" \
  "numba==0.65.1" \
  "llvmlite==0.47.0" \
  "bddl==1.0.1" \
  "gym==0.25.2" \
  "easydict>=1.9" \
  "future>=0.18.2" \
  "msgpack>=1.0" \
  "websockets>=12" \
  "pillow>=10" \
  "imageio[ffmpeg]>=2.34"

echo "[newil-headless] installing local editable packages without transitive deps"
conda run --no-capture-output -n "$NEWIL_CONDA_ENV" uv pip install --no-deps \
  -e third_party/robosuite \
  -e third_party/LIBERO \
  -e .

echo "[newil-headless] removing GUI OpenCV wheels if any resolver installed them"
conda run --no-capture-output -n "$NEWIL_CONDA_ENV" python -m pip uninstall -y \
  opencv-python \
  opencv-contrib-python \
  opencv-contrib-python-headless || true

echo "[newil-headless] re-pin numpy and headless OpenCV without dependency resolution"
conda run --no-capture-output -n "$NEWIL_CONDA_ENV" python -m pip install --force-reinstall --no-deps \
  "numpy==1.26.4" \
  "opencv-python-headless==4.10.0.84"

echo "[newil-headless] final package check"
conda run --no-capture-output -n "$NEWIL_CONDA_ENV" python - <<'PY'
import importlib.metadata as md
import sys

for forbidden in ("opencv-python", "opencv-contrib-python"):
    try:
        version = md.version(forbidden)
    except md.PackageNotFoundError:
        continue
    raise SystemExit(f"forbidden GUI OpenCV wheel is installed: {forbidden}=={version}")

for required in ("numpy", "opencv-python-headless", "mujoco", "robosuite", "libero"):
    print(f"{required}=={md.version(required)}")

import cv2
import numpy as np
print("cv2", cv2.__version__)
print("numpy", np.__version__)
PY

echo "[newil-headless] done"
