# New-IL Progress

## Current Goal

Build a small, public, reproducible LIBERO experiment that compares fixed-time
supervised fine-tuning against Progress-Aware Trajectory Cloud Supervision
(PA-TCS), while keeping this repository isolated from neighboring projects.

## Status

- [x] Read the New-IL idea documents and extracted the minimum experimental claim.
- [x] Added a `uv` project skeleton with Python 3.10 pinning.
- [x] Added a first NumPy implementation of PA-TCS phase clouds, tube loss, and
  progress metrics.
- [x] Added official-LIBERO-oriented data download and subset manifest entrypoints.
- [x] Added the first smoke experiment config and tests.
- [x] Downloaded one official LIBERO Object HDF5 smoke dataset under ignored
  `data/libero_small_hdf5/` and generated `data/libero_smoke_manifest.json`.
- [x] Added and ran OpenVLA modified LIBERO RLDS downloader for `spatial`,
  `object`, `goal`, and `long/libero_10`.
- [x] Added a guarded training launcher that records parameters, selected GPUs,
  terminal output, and status files.
- [x] Downloaded the four-suite OpenVLA modified LIBERO RLDS dataset under
  ignored `data/openvla_modified_libero_rlds/`:
  spatial 1.8G, object 2.7G, goal 1.8G, long/LIBERO-10 3.5G.
- [x] Added open-end DTW progress matching and gripper-event-contracted
  task-level olive trajectory clouds. Other demos define tube thickness; the
  strong distance is measured to the current sample's anchor trajectory so
  cross-demo progress timing is ignored without discarding task-completion time.
- [x] Added strong narrow event-channel loss for anchor gripper-transition nodes.
- [x] Changed olive cloud construction so anchor transition channels contract
  the task-level cloud after anchor and base cloud computation.
- [x] Added discrete phase polytope trajectory clouds: non-event sections are
  convex hulls with outward margin, query distances interpolate between phase
  sections, and event sections collapse directly to the anchor transition point.
- [x] Added `new-il-build-patcs-artifacts` to precompute phase clouds, anchors,
  event masks, gripper states, and padded hull equations from LIBERO HDF5 demos.
- [x] Upgraded memory planning to select up to two GPUs, choose 4-multiple
  per-device batches automatically, and reject risky manual batch sizes unless
  explicitly overridden.
- [x] Added intra-chunk and inter-chunk action smoothness losses.
- [x] Added project agent memory requiring subagents for independent parallel
  work by default.
- [x] Added a 3D olive trajectory-cloud visualization entrypoint.
- [x] Added PA-TCS artifact loader and online loss query layer
  (`src/new_il/training/patcs_loss.py`): loads precomputed hull equations,
  event masks, and anchors from `.npz` and computes tube loss + event channel
  loss per action chunk without re-running scipy at training time.
- [x] Wired LIBERO RLDS HDF5 data into a trainable action-chunk dataset with
  per-demo stage boundaries and rho_start precomputed from PATCS artifacts.
- [x] Implemented ActionMLPPolicy (small MLP fallback, ~82K params) with
  stable weight init; interface is swappable for SmolVLA later.
- [x] Implemented full action-expert trainer (train_ae.py) with:
  - BC loss: MSE on 7-dim action chunks
  - Differentiable PATCS loss: hull distance in PyTorch so gradients flow
    back through the predicted ee_pos trajectory (delta-xyz integration)
  - Gradient accumulation from run_guard env vars
  - CosineAnnealingLR, grad clip, per-step checkpointing with rotation
  - `new-il-train` entrypoint; `--patcs-weight 0.0` gives pure BC baseline
- [ ] Run fixed-time BC and PA-TCS on the same LIBERO subset.
- [ ] Add rollout evaluation in LIBERO simulation.

## Experiment Plan

1. For the very first data smoke test, use the downloaded single
   `libero_object` HDF5 task:
   `pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5`.
2. Then expand to `libero_spatial` with two tasks and up to ten demonstrations
   per task.
3. Use DTW to match generated/executed trajectory prefixes back to training
   trajectory progress, then choose the corresponding next observation index for
   the next generated chunk.
4. Train a small action-chunk policy with fixed-time BC:
   continuous action MSE plus gripper classification/regression.
5. Train the same policy with PA-TCS:
   olive trajectory cloud tube loss, contracted gripper-transition endpoints,
   anchor-distance supervision normalized by the task-level cloud thickness,
   strong event-channel constraints, and chunk smoothness regularization.
6. Compare mechanism metrics before relying on success rate:
   tube violation, event timing error, event pose error, wrong crossing, and
   progress backward rate.
7. Expand only after the smoke result is stable.

## Model Choice

The requested first model should stay below 1 GB. The best current public VLA
candidate for that constraint is `lerobot/smolvla_base` at roughly 907 MB. Public
LIBERO-finetuned OpenVLA checkpoints are strong references but are 7B/8B-class
models, so they do not satisfy the size constraint. If SmolVLA integration slows
the first ablation down, use a small ACT-style local policy as the controlled
fallback, then return to SmolVLA fine-tuning.

Useful small-model references:

- `lerobot/smolvla_base`: preferred public VLA candidate, around 907 MB.
- `lerobot/smolvla_libero` / `HuggingFaceVLA/smolvla_libero`: more LIBERO-ready
  but apparently slightly above 1 GB.
- `lerobot/vqbet_pusht`, ACT ALOHA checkpoints, and `lerobot/diffusion_pusht`:
  useful method baselines, but not LIBERO-native.

## External Sources Checked

- Official LIBERO GitHub documents dataset download through
  `benchmark_scripts/download_libero_datasets.py`, including specific suites such
  as `libero_spatial`, `libero_object`, `libero_goal`, and `libero_100`.
- Official LIBERO docs describe four core datasets and suite-specific download.
- OpenVLA Hugging Face model cards show public LIBERO-finetuned checkpoints, but
  they are 7B/8B-class and therefore not appropriate for the sub-1 GB smoke test.
- Subagent survey found `lerobot/smolvla_base` as the best sub-1 GB public VLA
  candidate and suggested starting from one official LIBERO Object HDF5 task.

## Isolation Rules

- Do not import from `../Mem` in New-IL code.
- Put external checkouts under `third_party/` and data/checkpoints under `data/`
  or `checkpoints/`, all ignored by git.
- Use `UV_CACHE_DIR=/tmp/uv-cache` for reproducible `uv` commands without
  polluting the repository.


## Time: 5/11 Problem:

现在的视频显示出这个行为模型只会原地张开夹爪不动，我怀疑是进度设计的问题，因为现在的进度设计没有强制监督的进度时间而是使用的DTW匹配，因此可能出现一直在起点进度附近卡loss bug，导致模型学会了原地张开夹爪的行为。建议在进度设计中加入强制监督的进度时间，或者调整DTW匹配的方式，以避免模型陷入这个局部最优解。 Eng: Video shows model only study dummy action in start position but no progress as usual, I suspect the progress design is the problem, because the current progress design does not have forced supervision progress time but uses DTW matching, so it may cause a loss bug that always stuck near the start progress, causing the model to learn the behavior of opening the gripper in place. It is recommended to add forced supervision progress time in the progress design, or adjust the way of DTW matching to avoid the model falling into this local optimal solution.

# 解决方法设想：

应该在前期给出较强的进度监督，以及减弱DIW给出的进度的参考（因为前期产生的抖动会导致DTW匹配不准确），训练到DTW的进度计算可以稳定的落在正确的进度区域内之后，再逐渐减弱强监督的进度时间，增加DTW匹配的权重，让模型逐渐适应DTW匹配的进度设计。这样可以避免模型在前期陷入局部最优解，同时也能让模型逐渐适应DTW匹配的进度设计。另外我们使用的模型是smolvla_base，其参数设置是aloha，但是我们测试的是libero，所以训练AE可以直接用smolvla_libero这个模型，在config和参数都适合。