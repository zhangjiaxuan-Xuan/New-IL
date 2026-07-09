# Image Rebuild Notes

Date: 2026-06-18

This snapshot was taken from the current New-IL container before rebuilding a local image.

## Conda Environments

Use these files as the primary references:

- `newil_environment.yml`
- `newil_conda_explicit.txt`
- `newil_pip_freeze.txt`
- `pi_environment.yml`
- `pi_conda_explicit.txt`
- `pi_pip_freeze.txt`

Recommended restore order:

```bash
conda env create -n newil -f newil_environment.yml
conda env create -n pi -f pi_environment.yml
```

If exact solve fails because the base image differs, recreate from the YAML first, then repair with the pip freeze files and editable installs:

```bash
conda run -n newil python -m pip install -r newil_pip_freeze.txt
conda run -n pi python -m pip install -r pi_pip_freeze.txt
```

## System EGL Packages

These system packages were installed without installing NVIDIA driver metapackages or kernel modules:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libegl1 \
  libglvnd0 \
  libglx0 \
  libopengl0 \
  libgl1 \
  libgles2 \
  libgbm1
```

Do not install these in the image unless intentionally rebuilding driver/kernel layers:

```text
nvidia-driver-*
nvidia-driver-*-open
linux-modules-nvidia-*
cuda-drivers
xserver-xorg-video-nvidia-*
```

## NVIDIA EGL Userspace

The container currently has NVIDIA 580.95.05 userspace EGL files prepared under:

```text
/data/L202500340/data/nvidia-egl-580/runfile-userspace-580.95.05
```

Those files were extracted from:

```text
/data/L202500340/data/nvidia-egl-580/NVIDIA-Linux-x86_64-580.95.05.run
```

The `.zshrc` block points to that root through:

```bash
export NVIDIA_EGL_ROOT=/data/L202500340/data/nvidia-egl-580/runfile-userspace-580.95.05
export LD_LIBRARY_PATH="$NVIDIA_EGL_ROOT/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="$NVIDIA_EGL_ROOT/usr/share/glvnd/egl_vendor.d"
export EGL_EXTERNAL_PLATFORM_CONFIG_DIRS="$NVIDIA_EGL_ROOT/usr/share/egl/egl_external_platform.d"
```

Important: in the current container, even with this userspace root, `eglQueryDevicesEXT()` still returned `0`. This means the rebuilt image should still be validated under the final runtime. If the runtime only exposes NVIDIA `compute,utility` capabilities, EGL may still fail until `graphics,display` are available.

## Cache And Data Defaults

The intended persistent data/cache roots are:

```bash
export NEWIL_DATA_ROOT=/data/L202500340/data
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/L202500340/data/huggingface
export HUGGINGFACE_HUB_CACHE=/data/L202500340/data/huggingface/hub
export TRANSFORMERS_CACHE=/data/L202500340/data/huggingface/transformers
export HF_DATASETS_CACHE=/data/L202500340/data/huggingface/datasets
export XDG_CACHE_HOME=/data/L202500340/data/.cache
export UV_CACHE_DIR=/data/L202500340/data/.cache/uv
export PIP_CACHE_DIR=/data/L202500340/data/.cache/pip
export OPENPI_CACHE_DIR=/data/L202500340/data/openpi
export OPENPI_CHECKPOINT_DIR=/data/L202500340/data/openpi/checkpoints
```

## Validation Commands

Import check:

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_EGL_DEVICE_ID=0 \
conda run --no-capture-output -n newil python - <<'PY'
from pathlib import Path
from new_il.libero.rollout import prepare_libero_paths
prepare_libero_paths(Path('/data/L202500340/New-IL/third_party/LIBERO'), Path('/tmp/libero_check'))
from libero.libero.envs import OffScreenRenderEnv
print('libero egl import ok')
PY
```

EGL device enumeration:

```bash
conda run --no-capture-output -n newil python - <<'PY'
from mujoco.egl import egl_ext as EGL
print('egl devices:', len(EGL.eglQueryDevicesEXT()))
PY
```

LIBERO rollout/render should only be considered valid if EGL devices is greater than zero and a real `OffScreenRenderEnv` reset/step produces nonblank RGB observations.
