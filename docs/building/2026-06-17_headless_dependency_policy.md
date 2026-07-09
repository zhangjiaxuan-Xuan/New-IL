# Headless Dependency Policy For Docker/POD Images

Time tag: 2026-06-17 17:35 UTC

This project must keep the New-IL runtime safe for Docker/POD images. The main
failure mode we hit was dependency resolution pulling GUI rendering wheels into
the image, especially `opencv-python`, which then required `libGL.so.1` and made
the image fragile or hard to save.

## Policy

For the `newil` environment:

- Use `opencv-python-headless`, not `opencv-python`.
- Do not install `opencv-contrib-python`.
- Do not switch LIBERO rollout to OSMesa for this project.
- Keep LIBERO/robosuite rendering on EGL.
- Keep OpenPI in the separate `pi` environment.
- Install local `robosuite`, `LIBERO`, and `new-il` with `--no-deps`.

The important point is the `--no-deps` editable install. The Mem robosuite fork
declares `opencv-python` as a dependency, so a normal editable install can pull a
GUI OpenCV wheel back into `newil`. We already install the compatible headless
runtime dependencies manually, so editable packages should not resolve their own
transitive deps.

## Safe Repair Command

From repo root:

```bash
cd /data/L202500340/New-IL
bash scripts/repair-newil-headless-env.sh
```

This script:

- installs only EGL/GLVND runtime libraries from conda-forge,
- pins `numpy==1.26.4`,
- pins `opencv-python-headless==4.10.0.84`,
- installs local editable packages with `--no-deps`,
- removes GUI OpenCV wheels if they were pulled in,
- verifies that `opencv-python` is not installed.

## Runtime EGL Variables

Use the Mem-proven EGL assignment:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=<worker_gpu_or_mig_id>
export MUJOCO_EGL_DEVICE_ID=0
```

`MUJOCO_EGL_DEVICE_ID=0` is intentional. With `CUDA_VISIBLE_DEVICES=i`, the
visible device is remapped to local device 0. The Mem robosuite fork already
contains the required assertion fix for this.

## Container Requirement

The container still must expose NVIDIA EGL devices. The headless Python
dependency policy cannot fix a missing NVIDIA EGL vendor mount.

A working container should satisfy:

```bash
ldconfig -p | grep -E 'libEGL_nvidia|libnvidia-eglcore'
find /usr/share/glvnd /etc/glvnd -name '*nvidia*.json'
```

and a render probe should report at least one EGL device. If EGL device count is
zero, LIBERO collection will fail before OpenPI or dynamic task assignment is
involved.

## Do Not Use For New-IL

Avoid these commands in the `newil` environment:

```bash
uv pip install -e third_party/robosuite
uv pip install -e third_party/LIBERO
uv pip install opencv-python
conda install mesalib libosmesa
```

Use the repair script instead.
