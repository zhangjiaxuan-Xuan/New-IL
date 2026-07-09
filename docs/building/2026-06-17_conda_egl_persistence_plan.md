# Conda EGL Persistence Plan

Time tag: 2026-06-17 18:20 UTC

## Background

The current Docker/POD image can run LIBERO EGL rendering after installing
system packages with `apt`, but those root-level changes may not persist when
the image is saved or restarted. The goal is to move as much of the headless EGL
runtime as possible into the `newil` conda environment and keep the root image
stable.

We should test this tomorrow using the older working image as the baseline.

## Current Findings

The New-IL Python dependency layer is now headless-safe:

- `opencv-python` absent.
- `opencv-contrib-python` absent.
- `opencv-python-headless==4.10.0.84`.
- `numpy==1.26.4`.
- editable `robosuite`, `LIBERO`, and `new-il` must be installed with
  `--no-deps`.

The runtime EGL issue is separate from Python dependencies:

- `libEGL`, `libGL`, `libOpenGL`, and Mesa/GLVND user-space libraries can be
  installed inside conda.
- NVIDIA driver EGL libraries such as `libEGL_nvidia.so` should normally come
  from the container runtime / host driver stack, because they must match the
  host driver.
- Installing random `libnvidia-gl-*` versions inside the container is risky if
  the apt package version does not match the host driver.

## Preferred Plan: Conda-First EGL

From repo root:

```bash
cd /data/L202500340/New-IL
```

Install stable headless GLVND/EGL runtime in `newil`:

```bash
conda install -y -n newil -c conda-forge \
  libegl=1.7.0 \
  libglvnd=1.7.0 \
  libgl=1.7.0 \
  libglx=1.7.0 \
  libopengl=1.7.0 \
  mesalib
```

Try adding Mesa EGL/GBM vendor packages into conda:

```bash
conda install -y -n newil -c conda-forge \
  mesa-libegl-conda-x86_64 \
  mesa-libgbm-conda-x86_64
```

Then keep Python dependencies headless:

```bash
bash scripts/repair-newil-headless-env.sh
```

Use conda library paths first during validation:

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/newil/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export NUMBA_CACHE_DIR=/tmp/newil_numba_cache
export MPLCONFIGDIR=/tmp/newil_mpl_cache
```

## Validation Commands

Import check:

```bash
conda run --no-capture-output -n newil python - <<'PY'
from pathlib import Path
from new_il.libero.rollout import prepare_libero_paths
prepare_libero_paths(Path("/data/L202500340/New-IL/third_party/LIBERO"), Path("/tmp/libero_check"))
from libero.libero.envs import OffScreenRenderEnv
print("libero egl import ok")
PY
```

Real render smoke:

```bash
cd /data/L202500340/New-IL
NUMBA_CACHE_DIR=/tmp/newil_numba_cache \
MPLCONFIGDIR=/tmp/newil_mpl_cache \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_EGL_DEVICE_ID=0 \
conda run --no-capture-output -n newil python - <<'PY'
from pathlib import Path
import json
import numpy as np
import torch

from new_il.libero.rollout import prepare_libero_paths, _format_lerobot_observation, LIBERO_DUMMY_ACTION

out = Path("runs/render_smoke_conda_egl")
out.mkdir(parents=True, exist_ok=True)
prepare_libero_paths(Path("third_party/LIBERO"), out)

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

suite = get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
states = torch.load(
    Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
    weights_only=False,
)

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl),
    camera_heights=256,
    camera_widths=256,
)
try:
    env.seed(7)
    env.set_init_state(states[0])
    obs = env.reset()
    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    formatted = _format_lerobot_observation(obs)
finally:
    env.close()

agent = np.asarray(obs["agentview_image"])
wrist = np.asarray(obs["robot0_eye_in_hand_image"])
summary = {
    "status": "ok",
    "agent_shape": list(agent.shape),
    "wrist_shape": list(wrist.shape),
    "agent_dtype": str(agent.dtype),
    "wrist_dtype": str(wrist.dtype),
    "agent_std": float(agent.std()),
    "wrist_std": float(wrist.std()),
    "policy_image_shape": list(formatted["pixels"]["image"].shape),
    "policy_wrist_shape": list(formatted["pixels"]["image2"].shape),
}
print(json.dumps(summary, indent=2))
PY
```

Pass criteria:

- `agent_shape == [256, 256, 3]`
- `wrist_shape == [256, 256, 3]`
- both dtypes are `uint8`
- both std values are nonzero
- no `OffScreenRenderEnv` EGL initialization error

## Fallback Plan: Dockerfile-Level Minimal Apt

If conda-only EGL does not survive without root apt packages, do not manually
apt install after container start. Put the minimal headless system packages into
the Dockerfile / image build:

```dockerfile
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libegl1 \
    libglvnd0 \
    libglx0 \
    libopengl0 \
    libgl1 \
    libgles2 \
    libgbm1 \
    libnvidia-egl-wayland1 \
    libnvidia-egl-gbm1 \
 && rm -rf /var/lib/apt/lists/*
```

Avoid installing `libnvidia-gl-580` unless we can pin a version compatible with
the host driver. The current host driver observed during debugging was
`580.95.05`, while apt exposed newer `580.159.03` NVIDIA GL packages. Mixing
these may work in some containers but is not the stable default.

## Do Not Do

Do not use these as normal setup steps:

```bash
uv pip install -e third_party/robosuite
uv pip install -e third_party/LIBERO
uv pip install opencv-python
conda install libosmesa
apt-get install libnvidia-gl-580
```

The first two can pull GUI OpenCV unless `--no-deps` is used. `libosmesa`
changes the rendering path and would require separate debugging. `libnvidia-gl`
inside the container risks driver/user-space mismatch.

## Next Step For Tomorrow

1. Boot the old working image.
2. Record its EGL library layout:

```bash
ldconfig -p | grep -E 'libEGL|libGLX|libOpenGL|libnvidia-egl|libnvidia-gl'
find /usr/share/glvnd /etc/glvnd -name '*.json' 2>/dev/null
```

3. Try the conda-first install commands above.
4. Run the real render smoke.
5. If render succeeds without apt, export a new `newil` conda snapshot.
6. If render needs system packages, bake the minimal apt package list into the
   Dockerfile instead of relying on runtime apt changes.
