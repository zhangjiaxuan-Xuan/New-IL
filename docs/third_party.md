# Third-Party Integration Notes

New-IL keeps external robotics stacks under `third_party/` or explicit symlinks
there. The directory itself is ignored by git; this file records the expected
layout and why each dependency exists.

## Current Layout

- `third_party/LIBERO`
  - Current local target: `/data/L202500340/Mem/third_party/LIBERO`
  - Purpose: LIBERO benchmark tasks, BDDL files, init states, and simulation eval.
  - Status command: `new-il-third-party-status`

- `third_party/robosuite-mem`
  - Current local target: `/data/L202500340/Mem/third_party/robosuite-mem`
  - Purpose: robosuite fork used by Mem. It includes EGL device-assignment fixes
    needed for multi-worker LIBERO collection.
  - Status command: `new-il-third-party-status`

- `third_party/robosuite`
  - Current local target: `/data/L202500340/Mem/third_party/robosuite-mem`
  - Purpose: package-name alias so `import robosuite` resolves to the Mem fork
    when `third_party/` is on `PYTHONPATH`.

- `third_party/openpi`
  - Current local checkout: `https://github.com/Physical-Intelligence/openpi.git`
  - Current commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
  - Purpose: OpenPI pi0/pi0.5 policy serving, fine-tuning, normalization stats,
    and LIBERO evaluation.
  - Environment note: upstream OpenPI requires Python `>=3.11`; the current
    `newil` conda environment is Python `3.10.20`. Treat OpenPI as a separate
    policy-serving environment unless New-IL is deliberately migrated to Python
    3.11.
  - LIBERO note: upstream OpenPI also has its own `third_party/libero` submodule
    path. New-IL should keep using the Mem-proven LIBERO and robosuite links for
    simulation, and use OpenPI primarily for policy loading/serving.

- `third_party/sdtw-cuda-torch`
  - Current local target: not present yet.
  - Purpose: optional reference implementation for differentiable Soft-DTW when
    moving PATCS from fixed chunk supervision to trajectory-level alignment.

## Immediate Policy

1. Use local symlinks for Mem-proven LIBERO and robosuite while New-IL is being
   split into a standalone project.
2. Keep external source code out of git.
3. Record exact source URLs, tags, and commits here once each checkout is
   installed.
4. Prefer upstream/OpenPI and LeRobot public APIs for policy loading and eval;
   only copy Mem code where it is New-IL-specific, especially rollout queueing
   and collection scheduling.

## Planned OpenPI Backend

The first OpenPI backend should target `pi05_libero` inference/eval before any
fine-tuning. Upstream OpenPI already provides the default server checkpoint:
`gs://openpi-assets/checkpoints/pi05_libero`.

1. Build an isolated Python 3.11 OpenPI environment.
2. Verify `openpi` import, JAX GPU visibility, and checkpoint access.
3. Add a `pi0`/`pi05` policy backend that can run either locally or through the
   OpenPI policy server. The New-IL-side websocket client lives in
   `new_il.integrations.openpi` and needs the `openpi-client` extra
   (`msgpack`, `pillow`, `websockets`) when it is used at runtime.
4. Run LIBERO one-task one-episode eval using New-IL/Mem LIBERO simulation and
   save action-health/video artifacts with
   `new-il-eval-openpi-libero-server`.
5. Only after eval is stable, add training and PATCS integration.

See `docs/openpi_newil_runtime.md` for the two-environment runtime contract and
parallel queue commands.
