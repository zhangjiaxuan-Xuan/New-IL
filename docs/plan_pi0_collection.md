# Plan: pi0/pi0.5 And Parallel Collection

This is the implementation plan for moving New-IL from the current MLP smoke
trainer toward pi0/pi0.5 plus dynamic continuous-observation collection.

## Phase 1: Third-Party Groundwork

- Create `third_party/` entries for LIBERO, robosuite-mem, OpenPI, and optional
  Soft-DTW reference code.
- Add `new-il-third-party-status` for local path and import checks.
- Keep OpenPI outside the current Python 3.10 training environment. Upstream
  OpenPI requires Python `>=3.11`, so the first integration should use a
  separate OpenPI policy server instead of forcing its dependencies into
  New-IL's existing conda environment.

## Phase 2: Policy Backend Interface

- Introduce a `PolicyBackend` interface with load, action inference, save, and
  eval adapter methods.
- Keep `ActionMLPPolicy` as a smoke backend.
- Add SmolVLA as the first VLA backend through LeRobot.
- Add pi0/pi0.5 as an OpenPI backend through `new_il.integrations.openpi`.
  Keep the New-IL rollout process as a lightweight websocket client; the heavy
  OpenPI model process should run in its Python 3.11 environment.

## Phase 3: OpenPI First-Pass Eval

- Use OpenPI's standard pi0.5 LIBERO checkpoint and policy server path.
- Server default: `scripts/serve_policy.py --env LIBERO`, which maps to
  `pi05_libero` at `gs://openpi-assets/checkpoints/pi05_libero`.
- Observation/action compatibility from upstream LIBERO example:
  - render at 256 pixels from LIBERO,
  - rotate agent and wrist images 180 degrees,
  - resize/pad to 224 pixels before policy inference,
  - provide `observation/image`, `observation/wrist_image`,
    `observation/state`, and `prompt`,
  - request 10-step action chunks and execute with 5-step replanning.
- Run one LIBERO task and one episode with saved video and action-health stats.
  New-IL entrypoint:
  `new-il-eval-openpi-libero-server --task-suite-name libero_spatial --task-id 0 --trials 1`.
- Confirm action normalization, observation schema, and image dimensions before
  any New-IL loss is attached.

## Phase 4: Mem-Style Parallel Collection

- Port the queue design from Mem, not the whole project:
  - `pending/`
  - `running/`
  - `ledger/done`
  - `ledger/success`
  - deficit scheduling with in-flight reservation
- Initial New-IL implementation:
  - `new-il-libero-openpi-make-queue`
  - `new-il-libero-openpi-rollout`
  - `new-il-libero-openpi-worker`
  - shell wrappers in `scripts/`
- Implement New-IL workers that can run SmolVLA or pi0.5. The pi0.5 path now
  uses the Mem-style canonical core in `new_il.libero.rollout`; only policy
  action generation is swapped to OpenPI websocket inference.
- Save collected trajectories as NPZ with:
  - raw render frames for reproducible video,
  - policy input frames after rotate/resize/pad,
  - `observation.images.image`
  - `observation.images.image2`
  - `observation.state`
  - `action`
  - `language`
  - `success`
  - metadata and action-health summaries

## Phase 5: Video And Health Checks

- Replace ad hoc video writing with one shared utility.
- Enforce `uint8`, `HWC`, contiguous frames, fixed frame size, and explicit FPS.
- Record action mean/std, xyz norm, gripper statistics, repeated-action rate,
  and all-zero checks for every rollout.

## Phase 6: PATCS Redesign

- Support artifact construction from Mem-style rollout NPZ directories with
  `new-il-build-patcs-artifacts --source-format rollout_npz`.
- First implement trajectory-level differentiable alignment:
  - Soft-DTW or monotonic alignment
  - anti-stall loss
  - backward-progress penalty
  - event loss with realistic radius and clipping
- Then implement the dynamic trajectory worker:
  - one worker owns one trajectory state
  - model output updates progress
  - next supervision chunk is selected from predicted progress
  - training continues until the trajectory reaches the terminal event
