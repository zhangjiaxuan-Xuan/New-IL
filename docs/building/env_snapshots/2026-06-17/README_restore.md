# Environment Snapshot Restore Notes

Snapshot time: 2026-06-17 17:00 UTC

This folder preserves the dependency state discovered during the New-IL / LIBERO / OpenPI setup work. Keep these files in git or copy them out of the image, because the current runtime image may not persist.

## Files

- `newil_environment.yml`: conda environment export for the New-IL main environment.
- `newil_conda_explicit.txt`: exact conda package URLs for `newil`.
- `newil_pip_freeze.txt`: pip package freeze for `newil`.
- `newil_uv_pip_freeze.txt`: uv package freeze for `newil`.
- `newil_import_check.txt`: successful import check for the New-IL LIBERO path.
- `pi_environment.yml`: conda environment export for the OpenPI environment.
- `pi_conda_explicit.txt`: exact conda package URLs for `pi`.
- `pi_pip_freeze.txt`: pip package freeze for `pi`.
- `pi_uv_pip_freeze.txt`: uv package freeze for `pi`.
- `pi_import_check.txt`: basic JAX import check for `pi`.

## Environment Roles

`newil` is the main project environment:

- Python 3.10.20
- LIBERO simulation
- robosuite / MuJoCo / EGL
- data conversion
- rollout queue workers
- video writing
- PATCS artifact construction
- SmolVLA integration work

`pi` is the OpenPI inference environment:

- Python 3.11.15
- OpenPI / pi0.5 model loading
- JAX inference
- websocket policy server

Do not merge these two environments. OpenPI should stay isolated and only serve policy inference.

## Preferred Restore

From repo root:

```bash
cd /data/L202500340/New-IL
conda env create -f docs/building/env_snapshots/2026-06-17/newil_environment.yml
conda env create -f docs/building/env_snapshots/2026-06-17/pi_environment.yml
```

If the conda solver behaves differently later, use the explicit files:

```bash
conda create -n newil --file docs/building/env_snapshots/2026-06-17/newil_conda_explicit.txt
conda create -n pi --file docs/building/env_snapshots/2026-06-17/pi_conda_explicit.txt
```

Then re-apply the project-local editable packages and cache paths:

```bash
conda activate newil
export UV_CACHE_DIR=/tmp/newil_uv_cache
bash scripts/repair-newil-headless-env.sh
```

```bash
conda activate pi
export UV_CACHE_DIR=/tmp/newil_uv_cache
uv pip install -e third_party/openpi
```

## Required Runtime Environment Variables

Use data/cache paths under `/data/L202500340/data` in real runs so downloads do not go into the image layer:

```bash
export NEW_IL_DATA_ROOT=/data/L202500340/data
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/L202500340/data/huggingface
export HF_HUB_CACHE=/data/L202500340/data/huggingface/hub
export HF_LEROBOT_HOME=/data/L202500340/data/huggingface/lerobot
export HF_DATASETS_CACHE=/data/L202500340/data/huggingface/datasets
export TRANSFORMERS_CACHE=/data/L202500340/data/huggingface/transformers
export OPENPI_DATA_HOME=/data/L202500340/data/openpi
export XDG_CACHE_HOME=/data/L202500340/data/cache
export UV_CACHE_DIR=/data/L202500340/data/uv-cache
```

For LIBERO / robosuite headless rendering:

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
```

For local import checks or sandboxed runs, these writable cache dirs may be needed:

```bash
export NUMBA_CACHE_DIR=/tmp/newil_numba_cache
export MPLCONFIGDIR=/tmp/newil_mpl_cache
export UV_CACHE_DIR=/tmp/newil_uv_cache
```

The `newil` import check only passed after setting `NUMBA_CACHE_DIR`; without it, local `third_party/robosuite` can fail during numba cache initialization.

## Important Dependency Notes

- `newil` currently uses `numpy==1.26.4`.
- `newil` currently uses `opencv-python-headless==4.10.0.84`; avoid replacing it with GUI OpenCV unless needed.
- `newil` currently uses `mujoco==3.9.0`.
- `newil` uses local `third_party/robosuite` and `third_party/LIBERO` paths during rollout.
- Install local `robosuite`, `LIBERO`, and `new-il` with `--no-deps` through `scripts/repair-newil-headless-env.sh`; a plain editable install can pull GUI `opencv-python` into the image.
- `pi` currently has `openpi==0.1.0` and `openpi-client==0.1.0`.
- The OpenPI checkpoint cache is expected at `/data/L202500340/data/openpi/openpi-assets/checkpoints/pi05_libero`.
- Installing `robomimic==0.2.0` attempted to build `egl-probe==1.0.2`, which failed with modern CMake policy errors. The current rollout path does not require robomimic.

## Verification Commands

New-IL:

```bash
cd /data/L202500340/New-IL
NUMBA_CACHE_DIR=/tmp/newil_numba_cache MPLCONFIGDIR=/tmp/newil_mpl_cache \
conda run --no-capture-output -n newil python -c "from pathlib import Path; from new_il.libero.rollout import prepare_libero_paths; prepare_libero_paths(Path('third_party/LIBERO'), Path('/tmp/libero_check')); import numpy, cv2, mujoco, robosuite; from libero.libero.envs import OffScreenRenderEnv; print('ok', numpy.__version__, cv2.__version__, mujoco.__version__)"
```

OpenPI:

```bash
conda run --no-capture-output -n pi python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Full test sanity:

```bash
conda run --no-capture-output -n newil pytest tests -q
```
