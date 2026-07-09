"""Action-expert trainer for LIBERO with optional PA-TCS supervision.

Reads per-device batch size and grad accumulation steps from environment
variables set by new-il-run (run_guard.py), or falls back to CLI args.

Loss design
-----------
  BC loss:    MSE(predicted_action_chunk, GT_action_chunk)  — all 7 dims
  PATCS loss: tube + event on predicted ee_pos trajectory
              predicted_ee_pos[k] = ee_pos[t] + cumsum(pred_delta_xyz[:k+1])

Usage
-----
  # via run_guard (recommended):
  UV_CACHE_DIR=/tmp/uv-cache uv run new-il-run \\
    --name smoke_object --max-gpus 2 \\
    -- python -m new_il.training.train_ae \\
         --suite object \\
         --hdf5-root data/libero_rlds_hdf5/object \\
         --artifact-root data/patcs_artifacts/object

  # direct (for debugging):
  UV_CACHE_DIR=/tmp/uv-cache uv run --extra train python -m new_il.training.train_ae \\
    --suite object \\
    --hdf5-root data/libero_rlds_hdf5/object \\
    --artifact-root data/patcs_artifacts/object \\
    --max-steps 500 --patcs-weight 0.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, random_split
    from torch.utils.data.distributed import DistributedSampler
except ImportError as exc:
    raise SystemExit("torch is required. Run: uv sync --extra train") from exc

from new_il.patcs import TubeLossConfig
from new_il.training.dataset import LiberoChunkDataset, RolloutChunkDataset, RolloutTrajectoryDataset
from new_il.training.model import ActionMLPPolicy
from new_il.training.patcs_loss import PatcsArtifact


# ---------------------------------------------------------------------------
# Progress curriculum: schedules for PATCS weight warmup and forced-progress decay
# ---------------------------------------------------------------------------

def _patcs_effective_weight(step: int, final_weight: float, warmup_steps: int) -> float:
    """Ramp PATCS weight linearly from 0 → final_weight over warmup_steps.

    During the warmup phase the model relies on BC loss to learn basic motion;
    once predictions are stable, PATCS provides full trajectory-cloud supervision.
    """
    if warmup_steps <= 0 or final_weight <= 0.0:
        return final_weight
    return final_weight * min(1.0, step / max(warmup_steps, 1))


def _progress_supervision_weight(step: int, initial_weight: float, decay_steps: int) -> float:
    """Decay direct-progress weight linearly from initial_weight → 0 over decay_steps.

    High early to prevent the 'stay-in-place + open-gripper' local optimum;
    decays to zero once the model has learned to advance through the task.
    """
    if decay_steps <= 0 or initial_weight <= 0.0:
        return 0.0
    return initial_weight * max(0.0, 1.0 - step / max(decay_steps, 1))


def _progress_forward_loss(
    pred_chunks: torch.Tensor,  # [B, H, 7]
    ee_pos_t: torch.Tensor,     # [B, 3]
    min_displacement: float,
) -> torch.Tensor:
    """Penalize predicted trajectories whose net end-effector displacement is too small.

    Integrates the predicted delta-xyz to get the predicted ee_pos trajectory,
    then measures ||pred_ee[-1] - ee_pos_t||.  Any sample whose total displacement
    is below min_displacement contributes a relu penalty, directly counteracting
    the 'gripper-open-in-place' local optimum that occurs during early training
    when DTW-based PATCS is unreliable.
    """
    pred_ee_last = ee_pos_t + torch.cumsum(pred_chunks[:, :, :3], dim=1)[:, -1, :]  # [B, 3]
    total_displacement = (pred_ee_last - ee_pos_t).norm(dim=-1)                      # [B]
    penalty = torch.relu(pred_chunks.new_tensor(min_displacement) - total_displacement)
    return penalty.mean()


def _temporal_consistency_loss(
    pred_chunks: torch.Tensor,       # [B, H, 7]
    ee_pos_seq: torch.Tensor,        # [B, H, 3]
    min_step_displacement: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Static proxy for dynamic PATCS progress constraints.

    Offline training cannot execute predicted chunks in the environment. This
    proxy uses the reference ee path to penalize per-step stalling and predicted
    xyz deltas that point backward relative to the local reference direction.
    """

    pred_xyz = pred_chunks[:, :, :3]
    pred_step_norm = pred_xyz.norm(dim=-1)
    stall = torch.relu(pred_chunks.new_tensor(min_step_displacement) - pred_step_norm).mean()

    if ee_pos_seq.shape[1] > 1:
        ref_delta = torch.diff(ee_pos_seq, dim=1)
        ref_delta = torch.cat([ref_delta, ref_delta[:, -1:, :]], dim=1)
        ref_norm = ref_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ref_dir = ref_delta / ref_norm
        backward = torch.relu(-(pred_xyz * ref_dir).sum(dim=-1)).mean()
    else:
        backward = pred_chunks.new_tensor(0.0)

    total = stall + backward
    return total, {
        "temporal_stall": stall.detach(),
        "temporal_backward": backward.detach(),
    }


def _smoothness_loss_torch(
    pred_chunks: torch.Tensor,       # [B, H, 7]
    prev_action: torch.Tensor,       # [B, 7]
    velocity_weight: float,
    acceleration_weight: float,
    boundary_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable intra-chunk and chunk-boundary smoothness."""

    zero = pred_chunks.new_tensor(0.0)
    if pred_chunks.shape[1] > 1:
        velocity = torch.diff(pred_chunks, dim=1).pow(2).mean()
    else:
        velocity = zero
    if pred_chunks.shape[1] > 2:
        acceleration = torch.diff(pred_chunks, n=2, dim=1).pow(2).mean()
    else:
        acceleration = zero
    boundary = (pred_chunks[:, 0, :] - prev_action).pow(2).mean()
    total = velocity_weight * velocity + acceleration_weight * acceleration + boundary_weight * boundary
    return total, {
        "smooth_velocity": velocity.detach(),
        "smooth_acceleration": acceleration.detach(),
        "smooth_boundary": boundary.detach(),
    }


def _action_health_metrics(pred_chunks: torch.Tensor) -> dict[str, torch.Tensor]:
    """Batch-level action diagnostics used to detect zero-action collapse."""

    with torch.no_grad():
        xyz_displacement = torch.cumsum(pred_chunks[:, :, :3], dim=1)[:, -1, :].norm(dim=-1)
        all_zero = (pred_chunks.abs().amax(dim=(1, 2)) < 1e-8).to(torch.float32)
        if pred_chunks.shape[1] > 1:
            repeated = (
                torch.diff(pred_chunks, dim=1).abs().amax(dim=(1, 2)) < 1e-6
            ).to(torch.float32)
        else:
            repeated = torch.zeros_like(all_zero)
        return {
            "action_mean_abs": pred_chunks.abs().mean().detach(),
            "action_std": pred_chunks.std().detach(),
            "xyz_displacement": xyz_displacement.mean().detach(),
            "all_zero_rate": all_zero.mean().detach(),
            "repeated_action_rate": repeated.mean().detach(),
        }


def _model_param_norm(model: nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                total += float(param.detach().pow(2).sum().item())
    return math.sqrt(total)


def _snapshot_trainable_params(model: nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


def _param_update_norm(model: nn.Module, before: list[torch.Tensor]) -> float:
    total = 0.0
    idx = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            total += float((param.detach() - before[idx]).pow(2).sum().item())
            idx += 1
    return math.sqrt(total)


def _health_warnings(prefix: str, metrics: dict[str, float | int], args: argparse.Namespace) -> list[str]:
    warnings = []
    if float(metrics.get("all_zero_rate", 0.0)) > args.health_max_all_zero_rate:
        warnings.append(f"{prefix}: all_zero_rate={metrics['all_zero_rate']:.4f}")
    if float(metrics.get("repeated_action_rate", 0.0)) > args.health_max_repeated_action_rate:
        warnings.append(f"{prefix}: repeated_action_rate={metrics['repeated_action_rate']:.4f}")
    if float(metrics.get("xyz_norm_mean", metrics.get("xyz_displacement", 1.0))) < args.health_min_xyz_norm:
        warnings.append(f"{prefix}: xyz_norm below threshold")
    if float(metrics.get("stage_oob_rate", 0.0)) > 0.0:
        warnings.append(f"{prefix}: stage_oob_rate={metrics['stage_oob_rate']:.4f}")
    if float(metrics.get("rho_oob_rate", 0.0)) > 0.0:
        warnings.append(f"{prefix}: rho_oob_rate={metrics['rho_oob_rate']:.4f}")
    return warnings


# ---------------------------------------------------------------------------
# Differentiable PATCS loss (PyTorch — gradients flow back to the model)
# ---------------------------------------------------------------------------

def _hull_section_loss_torch(
    point: torch.Tensor,      # [D]
    equations: torch.Tensor,  # [E, D+1]  halfspace rows
    count: int,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Softplus tube penalty for one phase cross-section.  Differentiable w.r.t. point."""
    if count <= 0:
        return point.new_tensor(0.0)
    eqs = equations[:count]                               # [count, D+1]
    violations = eqs[:, :-1] @ point + eqs[:, -1]        # [count]
    outside = torch.relu(violations.max() - margin)       # scalar ≥ 0
    margin_val = (outside - 1.0) / max(temperature, 1e-8)
    return torch.logaddexp(point.new_tensor(0.0), margin_val) ** 2


def _event_section_loss_torch(
    point: torch.Tensor,    # [D]
    anchor: torch.Tensor,   # [D]
    event_radius: float,
) -> torch.Tensor:
    """Normalized squared distance to anchor for an event phase. Differentiable."""
    normalized = (point - anchor) / max(event_radius, 1e-8)
    return (normalized ** 2).mean()


def _build_artifact_cache(artifact: PatcsArtifact, device: torch.device) -> dict:
    """Move only the arrays needed for GPU math to device. Control-flow arrays stay numpy."""
    return {
        # GPU tensors — used for differentiable ops only (no Python if-branches on these)
        "hull_equations_t": torch.as_tensor(artifact.hull_equations, device=device),  # [S,P,E,4]
        "anchor_t": torch.as_tensor(artifact.anchor, device=device),                  # [S,P,3]
        # CPU numpy — used for control flow (no GPU sync)
        "event_mask_np": artifact.event_mask,           # [S,P] bool
        "hull_counts_np": artifact.hull_equation_counts,  # [S,P] int32
        "phase_grid_np": artifact.phase_grid,           # [P] float32
    }


def _patcs_loss_batch(
    pred_chunks: torch.Tensor,     # [B, H, 7]
    ee_pos_t: torch.Tensor,        # [B, 3]
    stages: torch.Tensor,          # [B] int64
    rho_starts: torch.Tensor,      # [B] float32
    task_names: list[str],
    artifacts: dict[str, PatcsArtifact],
    artifact_cache: dict[str, dict],
    config: TubeLossConfig,
    event_weight: float,
    tube_weight: float,
    event_clip: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable PATCS loss with zero CPU-GPU synchronizations in the hot path.

    All control-flow decisions (which phase, event vs hull) use numpy arrays.
    GPU tensors are only touched for the actual floating-point math that needs
    gradients. This avoids the stall that was caused by evaluating GPU-tensor
    booleans in Python if-statements.

    Phase selection: single nearest phase per step (no window scan loop).
    Gradients flow through pred_chunks[:, :, :3] via the cumsum integration.
    """
    B, H = pred_chunks.shape[:2]

    # Predicted ee_pos trajectory: [B, H, 3] — differentiable
    pred_ee = ee_pos_t.unsqueeze(1) + torch.cumsum(pred_chunks[:, :, :3], dim=1)

    # Read stages / rhos onto CPU once (one sync, not per-sample-per-step)
    stages_np = stages.cpu().numpy()
    rhos_np = rho_starts.cpu().numpy()

    tube_terms: list[torch.Tensor] = []
    event_terms: list[torch.Tensor] = []

    for i in range(B):
        task = task_names[i]
        if task not in artifacts:
            continue
        artifact = artifacts[task]
        if task not in artifact_cache:
            artifact_cache[task] = _build_artifact_cache(artifact, device)
        cache = artifact_cache[task]

        stage = int(min(stages_np[i], artifact.num_stages - 1))
        rho = float(np.clip(rhos_np[i], 0.0, 1.0))
        phase_grid_np = cache["phase_grid_np"]          # numpy [P]
        event_mask_np = cache["event_mask_np"]          # numpy [S,P]
        hull_counts_np = cache["hull_counts_np"]        # numpy [S,P]
        P = len(phase_grid_np)

        for k in range(H):
            # pred_ee[k] is the position AFTER executing action k (i.e. at time t+k+1),
            # so the expected phase is rho_start + (k+1)*dt*v_max, not rho + k*dt*v_max.
            rho_k = float(np.clip(rho + (k + 1) * config.dt * config.v_max, 0.0, 1.0))
            p_idx = int(np.round(rho_k * (P - 1)))

            point = pred_ee[i, k]   # [3], on GPU, has grad

            # All control-flow uses numpy (zero GPU sync)
            if bool(event_mask_np[stage, p_idx]):
                # Event phase: MSE to anchor
                anchor = cache["anchor_t"][stage, p_idx]            # [3] GPU
                d = ((point - anchor) / max(artifact.event_radius, 1e-8)).pow(2).mean()
                if event_clip > 0.0:
                    d = torch.clamp(d, max=event_clip)
                event_terms.append(d)
            else:
                count = int(hull_counts_np[stage, p_idx])           # numpy int
                if count > 0:
                    eqs = cache["hull_equations_t"][stage, p_idx, :count]  # [count,4] GPU
                    violations = eqs[:, :3] @ point + eqs[:, 3]            # [count]
                    outside = torch.relu(violations.max() - artifact.margin)
                    mv = (outside - 1.0) / max(config.temperature, 1e-8)
                    tube_terms.append(torch.logaddexp(point.new_tensor(0.0), mv) ** 2)

    zero = pred_chunks.new_tensor(0.0)
    tube_t = torch.stack(tube_terms).mean() if tube_terms else zero
    event_t = torch.stack(event_terms).mean() if event_terms else zero
    total_t = tube_weight * tube_t + event_weight * event_t

    return total_t, {
        "patcs_total": total_t.detach(),
        "patcs_tube": tube_t.detach(),
        "patcs_event": event_t.detach(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _cosine_warmup_lr_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """LR multiplier: linear warmup then cosine decay.

    Returns a value in [min_lr_ratio, 1.0] to be multiplied by the base lr.
      step < warmup_steps  → linearly ramp 0 → 1
      step >= warmup_steps → cosine decay 1 → min_lr_ratio
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _env_int(key: str, fallback: int) -> int:
    val = os.environ.get(key)
    return int(val) if val and val.isdigit() else fallback


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed() -> tuple[bool, int, int, int]:
    distributed = _is_distributed()
    if not distributed:
        return False, 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return True, local_rank, rank, world_size


def _cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _print_rank0(rank: int, *args, **kwargs) -> None:
    if rank == 0:
        print(*args, **kwargs)


def _barrier(distributed: bool, device: torch.device) -> None:
    if not distributed:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _metric_means(
    log_metrics: dict[str, list[torch.Tensor]],
    distributed: bool,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    means: dict[str, float] = {}
    for key, values in log_metrics.items():
        if values:
            value = torch.stack(values).mean()
        else:
            value = torch.tensor(0.0, device=device)
        if distributed:
            value = value.clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= world_size
        means[key] = float(value.item())
    return means


def _sync_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    distributed: bool,
    rank: int,
    device: torch.device,
) -> None:
    if not distributed:
        return
    values = [group["lr"] for group in optimizer.param_groups] if rank == 0 else [0.0] * len(optimizer.param_groups)
    lr_tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.broadcast(lr_tensor, src=0)
    for group, lr in zip(optimizer.param_groups, lr_tensor.tolist()):
        group["lr"] = float(lr)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _init_swanlab(
    *,
    enabled: bool,
    project: str,
    run_name: str | None,
    mode: str,
    output_dir: Path,
    config: dict,
):
    if not enabled:
        return None
    try:
        import swanlab  # noqa: PLC0415
    except ImportError:
        print("SwanLab is not installed; continuing without SwanLab logging.", flush=True)
        return None
    try:
        return swanlab.init(
            project=project,
            experiment_name=run_name,
            mode=mode,
            logdir=str(output_dir),
            config=config,
        )
    except Exception as exc:
        print(f"SwanLab init failed: {exc}", flush=True)
        return None


def _log_swanlab(run, record: dict, step: int) -> None:
    if run is None:
        return
    try:
        import swanlab  # noqa: PLC0415

        swanlab.log(record, step=step)
    except Exception as exc:
        print(f"SwanLab logging failed at step {step}: {exc}", flush=True)


def _finish_swanlab(run) -> None:
    if run is None:
        return
    try:
        import swanlab  # noqa: PLC0415

        swanlab.finish()
    except Exception as exc:
        print(f"SwanLab finish failed: {exc}", flush=True)


def _dynamic_batch(dataset: RolloutTrajectoryDataset, worker_states: list) -> dict:
    items = [dataset.chunk_at(state) for state in worker_states]
    return {
        "obs": torch.as_tensor(np.stack([item["obs"] for item in items]), dtype=torch.float32),
        "action_chunk": torch.as_tensor(
            np.stack([item["action_chunk"] for item in items]), dtype=torch.float32
        ),
        "ee_pos_seq": torch.as_tensor(np.stack([item["ee_pos_seq"] for item in items]), dtype=torch.float32),
        "prev_action": torch.as_tensor(np.stack([item["prev_action"] for item in items]), dtype=torch.float32),
        "stage": torch.as_tensor([item["stage"] for item in items], dtype=torch.int64),
        "rho_start": torch.as_tensor([item["rho_start"] for item in items], dtype=torch.float32),
        "task_name": [item["task_name"] for item in items],
        "obs_index": np.asarray([item["obs_index"] for item in items], dtype=np.int32),
    }


def _run_dynamic_trajectory_training(
    args: argparse.Namespace,
    *,
    device: torch.device,
    rank: int,
    distributed: bool,
    per_device_batch: int,
    grad_accum: int,
) -> None:
    """Simplified dynamic PA-TCS loop for rollout trajectories.

    This is the first executable approximation of the target Trajectory Worker
    algorithm. The policy prediction determines each worker's next observation
    index by monotonic nearest-neighbor matching against the same demo ee path.
    """

    if distributed:
        raise RuntimeError("--dynamic-trajectory currently supports single-process training only.")
    if args.rollout_root is None:
        raise ValueError("--dynamic-trajectory requires --rollout-root.")

    dataset = RolloutTrajectoryDataset(
        rollout_root=args.rollout_root,
        artifact_dir=args.artifact_root,
        horizon=args.action_horizon,
        obs_full_state=args.obs_full_state,
        max_tasks=args.max_tasks,
        verbose=(rank == 0),
    )
    data_health = dataset.health_summary()
    warnings: list[str] = _health_warnings("data", data_health, args)
    _print_rank0(rank, f"Dynamic data health: {data_health}", flush=True)
    for warning in warnings:
        _print_rank0(rank, f"HEALTH WARNING: {warning}", flush=True)

    model = ActionMLPPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        horizon=args.action_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    _print_rank0(rank, f"Model: ActionMLPPolicy  params={model.num_parameters():,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_lr_ratio = args.min_lr / max(args.lr, 1e-12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup_lr_lambda(
            step, args.lr_warmup_steps, args.max_steps, min_lr_ratio
        ),
    )
    tube_config = TubeLossConfig(
        sigma=args.tube_sigma,
        temperature=args.tube_temperature,
        v_min=0.0,
        v_max=args.tube_v_max,
        delta=args.tube_delta,
        dt=1.0 / args.action_horizon,
    )
    artifact_cache: dict[str, dict] = {}
    ckpt_dir = Path(args.checkpoint_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    worker_count = args.dynamic_workers or per_device_batch
    workers = dataset.initial_worker_states(worker_count, args.seed)
    mse_loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    accum_count = 0
    step = 0
    log_metrics: dict[str, list[torch.Tensor]] = {
        "loss": [], "bc_loss": [], "patcs_loss": [], "patcs_tube": [], "patcs_event": [],
        "progress_loss": [], "temporal_loss": [], "temporal_stall": [], "temporal_backward": [],
        "smoothness_loss": [], "action_mean_abs": [], "action_std": [], "xyz_displacement": [],
        "all_zero_rate": [], "repeated_action_rate": [], "dynamic_rho_advance": [],
        "dynamic_stall_rate": [], "dynamic_reset_rate": [],
        "dynamic_matcher_cost": [], "dynamic_matcher_used": [],
        "grad_norm": [], "param_update_ratio": [],
    }
    last_mean: dict[str, float] = {}
    t0 = time.time()
    _print_rank0(
        rank,
        f"Starting dynamic trajectory training: max_steps={args.max_steps} workers={worker_count} "
        f"patcs_weight={args.patcs_weight}",
        flush=True,
    )

    while step < args.max_steps:
        batch = _dynamic_batch(dataset, workers)
        obs = batch["obs"].to(device)
        action_gt = batch["action_chunk"].to(device)
        ee_pos_seq = batch["ee_pos_seq"].to(device)
        ee_pos_t = ee_pos_seq[:, 0, :]
        prev_action = batch["prev_action"].to(device)

        pred = model(obs)
        bc = mse_loss_fn(pred, action_gt)
        patcs_w_eff = _patcs_effective_weight(step, args.patcs_weight, args.patcs_warmup_steps)
        prog_w_eff = _progress_supervision_weight(
            step, args.progress_supervision_weight, args.progress_supervision_decay_steps
        )

        patcs_metrics: dict[str, torch.Tensor] = {}
        patcs_scalar = torch.tensor(0.0, device=device)
        if patcs_w_eff > 0.0:
            patcs_scalar, patcs_metrics = _patcs_loss_batch(
                pred,
                ee_pos_t,
                batch["stage"],
                batch["rho_start"],
                batch["task_name"],
                dataset.artifacts,
                artifact_cache,
                tube_config,
                event_weight=args.event_weight,
                tube_weight=1.0,
                event_clip=args.event_clip,
                device=device,
            )

        prog_loss = torch.tensor(0.0, device=device)
        if prog_w_eff > 0.0:
            prog_loss = _progress_forward_loss(pred, ee_pos_t, args.progress_min_displacement)

        temporal_loss = torch.tensor(0.0, device=device)
        temporal_metrics: dict[str, torch.Tensor] = {}
        if args.temporal_consistency_weight > 0.0:
            temporal_loss, temporal_metrics = _temporal_consistency_loss(
                pred, ee_pos_seq, args.temporal_min_step_displacement
            )

        smoothness_loss = torch.tensor(0.0, device=device)
        if args.smoothness_weight > 0.0:
            smoothness_loss, _ = _smoothness_loss_torch(
                pred,
                prev_action,
                velocity_weight=args.smooth_velocity_weight,
                acceleration_weight=args.smooth_acceleration_weight,
                boundary_weight=args.smooth_boundary_weight,
            )

        loss = (
            args.bc_weight * bc
            + patcs_w_eff * patcs_scalar
            + prog_w_eff * prog_loss
            + args.temporal_consistency_weight * temporal_loss
            + args.smoothness_weight * smoothness_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite dynamic training loss at step {step}: {loss.detach().cpu().item()}")
        (loss / grad_accum).backward()
        accum_count += 1

        pred_np = pred.detach().cpu().numpy()
        advance_stats = []
        for idx, worker in enumerate(workers):
            next_worker, stats = dataset.advance_worker_state(
                worker,
                pred_np[idx],
                min_advance=args.dynamic_min_advance,
                max_advance=args.dynamic_max_advance or args.action_horizon,
                matcher=args.dynamic_matcher,
                match_cost=args.dynamic_match_cost,
            )
            workers[idx] = next_worker
            advance_stats.append(stats)

        zero_metric = loss.detach().new_tensor(0.0)
        health = _action_health_metrics(pred.detach())
        log_metrics["loss"].append(loss.detach())
        log_metrics["bc_loss"].append(bc.detach())
        log_metrics["patcs_loss"].append(patcs_scalar.detach())
        log_metrics["patcs_tube"].append(patcs_metrics.get("patcs_tube", zero_metric))
        log_metrics["patcs_event"].append(patcs_metrics.get("patcs_event", zero_metric))
        log_metrics["progress_loss"].append(prog_loss.detach())
        log_metrics["temporal_loss"].append(temporal_loss.detach())
        log_metrics["temporal_stall"].append(temporal_metrics.get("temporal_stall", zero_metric))
        log_metrics["temporal_backward"].append(temporal_metrics.get("temporal_backward", zero_metric))
        log_metrics["smoothness_loss"].append(smoothness_loss.detach())
        for key, value in health.items():
            log_metrics[key].append(value)
        log_metrics["dynamic_rho_advance"].append(
            loss.detach().new_tensor(float(np.mean([s["rho_advance"] for s in advance_stats])))
        )
        log_metrics["dynamic_stall_rate"].append(
            loss.detach().new_tensor(float(np.mean([s["stall"] for s in advance_stats])))
        )
        log_metrics["dynamic_reset_rate"].append(
            loss.detach().new_tensor(float(np.mean([s["reset"] for s in advance_stats])))
        )
        log_metrics["dynamic_matcher_cost"].append(
            loss.detach().new_tensor(float(np.mean([s["matcher_cost"] for s in advance_stats])))
        )
        log_metrics["dynamic_matcher_used"].append(
            loss.detach().new_tensor(float(np.mean([s["matcher_used"] for s in advance_stats])))
        )

        if accum_count >= grad_accum:
            params_before = _snapshot_trainable_params(model)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise RuntimeError(f"Non-finite grad norm at step {step}: {grad_norm}")
            param_norm_before = max(_model_param_norm(model), 1e-12)
            optimizer.step()
            update_norm = _param_update_norm(model, params_before)
            update_ratio = update_norm / param_norm_before
            if not math.isfinite(update_ratio):
                raise RuntimeError(f"Non-finite parameter update ratio at step {step}: {update_ratio}")
            if update_ratio > args.health_max_update_ratio:
                warning = f"train: param_update_ratio={update_ratio:.6f}"
                warnings.append(warning)
                _print_rank0(rank, f"HEALTH WARNING: {warning}", flush=True)
            log_metrics["grad_norm"].append(loss.detach().new_tensor(float(grad_norm)))
            log_metrics["param_update_ratio"].append(loss.detach().new_tensor(float(update_ratio)))
            scheduler.step()
            optimizer.zero_grad()
            accum_count = 0
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                mean = _metric_means(log_metrics, False, 1, device)
                last_mean = mean
                warnings.extend(_health_warnings("pred", mean, args))
                if mean["dynamic_stall_rate"] > args.health_max_stall_rate:
                    warnings.append(f"dynamic: stall_rate={mean['dynamic_stall_rate']:.4f}")
                _print_rank0(
                    rank,
                    f"dyn_step={step:>6d}/{args.max_steps} loss={mean['loss']:.4f} "
                    f"bc={mean['bc_loss']:.4f} patcs={mean['patcs_loss']:.4f} "
                    f"rho_adv={mean['dynamic_rho_advance']:.4f} "
                    f"stall={mean['dynamic_stall_rate']:.3f} reset={mean['dynamic_reset_rate']:.3f} "
                    f"match={mean['dynamic_matcher_cost']:.4f} "
                    f"|a|={mean['action_mean_abs']:.4f} xyz={mean['xyz_displacement']:.4f} "
                    f"grad={mean['grad_norm']:.4f} upd={mean['param_update_ratio']:.2e} "
                    f"zero={mean['all_zero_rate']:.3f} repeat={mean['repeated_action_rate']:.3f} "
                    f"{elapsed:.0f}s",
                    flush=True,
                )
                for values in log_metrics.values():
                    values.clear()
                t0 = time.time()

            if rank == 0 and (step % args.checkpoint_every == 0 or step == args.max_steps):
                checkpoint_path = _save_checkpoint(model, optimizer, scheduler, step, args, ckpt_dir)
                _rotate_checkpoints(ckpt_dir, keep=args.keep_last_checkpoints)
                print(f"Dynamic checkpoint: {checkpoint_path}", flush=True)

    if rank == 0:
        summary = {
            "mode": "dynamic_trajectory",
            "final_step": step,
            "worker_count": worker_count,
            "num_trajectories": len(dataset.trajectories),
            "dynamic_matcher": args.dynamic_matcher,
            "dynamic_match_cost": args.dynamic_match_cost,
            "data_health": data_health,
            "last_train_health": last_mean,
            "health_warnings": sorted(set(warnings)),
            "args": _json_safe(vars(args)),
        }
        (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Dynamic training complete. Checkpoints in {ckpt_dir}", flush=True)


def train(args: argparse.Namespace) -> None:
    distributed, local_rank, rank, world_size = _setup_distributed()
    swanlab_run = None
    # --- resolve batch / accumulation from env (set by run_guard) or CLI ---
    per_device_batch = _env_int("NEW_IL_PER_DEVICE_BATCH_SIZE", args.per_device_batch_size)
    grad_accum = _env_int("NEW_IL_GRAD_ACCUMULATION_STEPS", args.grad_accumulation_steps)

    device = torch.device(
        f"cuda:{local_rank}" if distributed and torch.cuda.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    _print_rank0(
        rank,
        f"Device: {device}  distributed={distributed}  world_size={world_size}  "
        f"per_device_batch={per_device_batch}  grad_accum={grad_accum}",
        flush=True,
    )

    try:
        if args.dynamic_trajectory:
            _run_dynamic_trajectory_training(
                args,
                device=device,
                rank=rank,
                distributed=distributed,
                per_device_batch=per_device_batch,
                grad_accum=grad_accum,
            )
            return

        # --- dataset ---
        if args.rollout_root is not None:
            full_dataset = RolloutChunkDataset(
                rollout_root=args.rollout_root,
                artifact_dir=args.artifact_root,
                horizon=args.action_horizon,
                obs_full_state=args.obs_full_state,
                max_tasks=args.max_tasks,
                verbose=(rank == 0),
            )
        else:
            if args.hdf5_root is None:
                raise ValueError("Either --hdf5-root or --rollout-root is required.")
            full_dataset = LiberoChunkDataset(
                hdf5_dir=args.hdf5_root,
                artifact_dir=args.artifact_root,
                horizon=args.action_horizon,
                obs_full_state=args.obs_full_state,
                max_tasks=args.max_tasks,
                verbose=(rank == 0),
            )
        health_warnings: list[str] = []
        if hasattr(full_dataset, "health_summary"):
            data_health = full_dataset.health_summary()
            _print_rank0(rank, f"Data health: {data_health}", flush=True)
            health_warnings.extend(_health_warnings("data", data_health, args))
            for warning in health_warnings:
                _print_rank0(rank, f"HEALTH WARNING: {warning}", flush=True)
        else:
            data_health = {}
        val_size = max(1, int(len(full_dataset) * args.val_fraction))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        ) if distributed else None
        train_loader = DataLoader(
            train_ds,
            batch_size=per_device_batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=per_device_batch * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # --- model ---
        model = ActionMLPPolicy(
            obs_dim=full_dataset.obs_dim,
            action_dim=full_dataset.action_dim,
            horizon=args.action_horizon,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
        _print_rank0(rank, f"Model: ActionMLPPolicy  params={_unwrap_model(model).num_parameters():,}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = args.max_steps
        min_lr_ratio = args.min_lr / max(args.lr, 1e-12)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_warmup_lr_lambda(
                step, args.lr_warmup_steps, total_steps, min_lr_ratio
            ),
        )

        # --- PATCS config ---
        tube_config = TubeLossConfig(
            sigma=args.tube_sigma,
            temperature=args.tube_temperature,
            v_min=0.0,
            v_max=args.tube_v_max,
            delta=args.tube_delta,
            dt=1.0 / args.action_horizon,
        )
        # Pre-collect all artifacts; cache will hold device tensors (populated lazily)
        all_artifacts: dict[str, PatcsArtifact] = full_dataset._artifacts  # noqa: SLF001
        artifact_cache: dict[str, dict] = {}

        # --- checkpoint dir ---
        ckpt_dir = Path(args.checkpoint_dir)
        if rank == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        _barrier(distributed, device)

        swanlab_run = _init_swanlab(
            enabled=args.swanlab and rank == 0,
            project=args.swanlab_project,
            run_name=args.swanlab_run_name or f"{args.suite}_patcs_ae",
            mode=args.swanlab_mode,
            output_dir=args.swanlab_logdir or (ckpt_dir / "swanlab"),
            config={
                "suite": args.suite,
                "hdf5_root": str(args.hdf5_root) if args.hdf5_root is not None else None,
                "rollout_root": str(args.rollout_root) if args.rollout_root is not None else None,
                "artifact_root": str(args.artifact_root),
                "checkpoint_dir": str(args.checkpoint_dir),
                "max_steps": args.max_steps,
                "per_device_batch": per_device_batch,
                "grad_accum": grad_accum,
                "world_size": world_size,
                "effective_batch_size": per_device_batch * grad_accum * world_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "bc_weight": args.bc_weight,
                "patcs_weight": args.patcs_weight,
                "event_weight": args.event_weight,
                "event_clip": args.event_clip,
                "patcs_warmup_steps": args.patcs_warmup_steps,
                "progress_supervision_weight": args.progress_supervision_weight,
                "progress_supervision_decay_steps": args.progress_supervision_decay_steps,
                "progress_min_displacement": args.progress_min_displacement,
                "temporal_consistency_weight": args.temporal_consistency_weight,
                "temporal_min_step_displacement": args.temporal_min_step_displacement,
                "smoothness_weight": args.smoothness_weight,
                "smooth_velocity_weight": args.smooth_velocity_weight,
                "smooth_acceleration_weight": args.smooth_acceleration_weight,
                "smooth_boundary_weight": args.smooth_boundary_weight,
                "action_horizon": args.action_horizon,
                "obs_full_state": args.obs_full_state,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "lr_warmup_steps": args.lr_warmup_steps,
                "min_lr": args.min_lr,
                "eval_on_checkpoint": args.eval_on_checkpoint,
                "eval_server_url": args.eval_server_url,
                "eval_benchmark": args.eval_benchmark or _libero_benchmark_name(args.suite),
                "eval_task_id": args.eval_task_id,
                "eval_trials": args.eval_trials,
            },
        )

        # --- training loop ---
        mse_loss_fn = nn.MSELoss()
        step = 0
        optimizer.zero_grad()
        accum_count = 0
        log_metrics: dict[str, list[torch.Tensor]] = {
            "loss": [], "bc_loss": [], "patcs_loss": [],
            "patcs_tube": [], "patcs_event": [],
            "progress_loss": [], "temporal_loss": [],
            "temporal_stall": [], "temporal_backward": [],
            "smoothness_loss": [], "smooth_velocity": [],
            "smooth_acceleration": [], "smooth_boundary": [],
            "action_mean_abs": [], "action_std": [], "xyz_displacement": [],
            "all_zero_rate": [], "repeated_action_rate": [],
            "grad_norm": [], "param_update_ratio": [],
        }
        last_train_health: dict[str, float] = {}

        _print_rank0(
            rank,
            f"Starting training: max_steps={total_steps}  patcs_weight={args.patcs_weight}  "
            f"patcs_warmup_steps={args.patcs_warmup_steps}  "
            f"progress_supervision_weight={args.progress_supervision_weight}  "
            f"progress_supervision_decay_steps={args.progress_supervision_decay_steps}",
            flush=True,
        )
        t0 = time.time()

        epoch = 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        while step < total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            obs = batch["obs"].to(device, dtype=torch.float32)              # [B, obs_dim]
            action_gt = batch["action_chunk"].to(device, dtype=torch.float32)  # [B, H, 7]
            ee_pos_seq = batch["ee_pos_seq"].to(device, dtype=torch.float32)  # [B, H, 3]
            ee_pos_t = ee_pos_seq[:, 0, :]  # [B, 3]
            prev_action = batch["prev_action"].to(device, dtype=torch.float32)  # [B, 7]

            pred = model(obs)                                                # [B, H, 7]

            # BC loss
            bc = mse_loss_fn(pred, action_gt)

            # Curriculum schedules — computed per optimizer step so micro-batches
            # within an accumulation window share the same effective weights.
            patcs_w_eff = _patcs_effective_weight(step, args.patcs_weight, args.patcs_warmup_steps)
            prog_w_eff = _progress_supervision_weight(
                step, args.progress_supervision_weight, args.progress_supervision_decay_steps
            )

            # PATCS loss with warmup-ramped weight — negligible early so noisy
            # DTW-based phase supervision does not mislead the model before it
            # has learned basic motion.
            patcs_metrics: dict[str, torch.Tensor] = {}
            patcs_scalar = torch.tensor(0.0, device=device)
            if patcs_w_eff > 0.0:
                patcs_scalar, patcs_metrics = _patcs_loss_batch(
                    pred,
                    ee_pos_t,
                    batch["stage"],
                    batch["rho_start"],
                    batch["task_name"],
                    all_artifacts,
                    artifact_cache,
                    tube_config,
                    event_weight=args.event_weight,
                    tube_weight=1.0,
                    event_clip=args.event_clip,
                    device=device,
                )

            # Direct progress-forward loss — high early to prevent the
            # 'open-gripper-in-place' local optimum; decays once the model
            # has learned to advance through the trajectory.
            prog_loss = torch.tensor(0.0, device=device)
            if prog_w_eff > 0.0:
                prog_loss = _progress_forward_loss(pred, ee_pos_t, args.progress_min_displacement)

            temporal_loss = torch.tensor(0.0, device=device)
            temporal_metrics: dict[str, torch.Tensor] = {}
            if args.temporal_consistency_weight > 0.0:
                temporal_loss, temporal_metrics = _temporal_consistency_loss(
                    pred, ee_pos_seq, args.temporal_min_step_displacement
                )

            smoothness_loss = torch.tensor(0.0, device=device)
            smooth_metrics: dict[str, torch.Tensor] = {}
            if args.smoothness_weight > 0.0:
                smoothness_loss, smooth_metrics = _smoothness_loss_torch(
                    pred,
                    prev_action,
                    velocity_weight=args.smooth_velocity_weight,
                    acceleration_weight=args.smooth_acceleration_weight,
                    boundary_weight=args.smooth_boundary_weight,
                )

            loss = (
                args.bc_weight * bc
                + patcs_w_eff * patcs_scalar
                + prog_w_eff * prog_loss
                + args.temporal_consistency_weight * temporal_loss
                + args.smoothness_weight * smoothness_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}: {loss.detach().cpu().item()}")
            (loss / grad_accum).backward()
            accum_count += 1

            # Accumulate detached tensors and synchronize only when logging.
            zero_metric = loss.detach().new_tensor(0.0)
            health = _action_health_metrics(pred.detach())
            log_metrics["loss"].append(loss.detach())
            log_metrics["bc_loss"].append(bc.detach())
            log_metrics["patcs_loss"].append(patcs_scalar.detach())
            log_metrics["patcs_tube"].append(patcs_metrics.get("patcs_tube", zero_metric))
            log_metrics["patcs_event"].append(patcs_metrics.get("patcs_event", zero_metric))
            log_metrics["progress_loss"].append(prog_loss.detach())
            log_metrics["temporal_loss"].append(temporal_loss.detach())
            log_metrics["temporal_stall"].append(temporal_metrics.get("temporal_stall", zero_metric))
            log_metrics["temporal_backward"].append(temporal_metrics.get("temporal_backward", zero_metric))
            log_metrics["smoothness_loss"].append(smoothness_loss.detach())
            log_metrics["smooth_velocity"].append(smooth_metrics.get("smooth_velocity", zero_metric))
            log_metrics["smooth_acceleration"].append(smooth_metrics.get("smooth_acceleration", zero_metric))
            log_metrics["smooth_boundary"].append(smooth_metrics.get("smooth_boundary", zero_metric))
            for key, value in health.items():
                log_metrics[key].append(value)

            if accum_count >= grad_accum:
                params_before = _snapshot_trainable_params(_unwrap_model(model))
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError(f"Non-finite grad norm at step {step}: {grad_norm}")
                param_norm_before = max(_model_param_norm(_unwrap_model(model)), 1e-12)
                optimizer.step()
                update_norm = _param_update_norm(_unwrap_model(model), params_before)
                update_ratio = update_norm / param_norm_before
                if not math.isfinite(update_ratio):
                    raise RuntimeError(f"Non-finite parameter update ratio at step {step}: {update_ratio}")
                if update_ratio > args.health_max_update_ratio:
                    warning = f"train: param_update_ratio={update_ratio:.6f}"
                    health_warnings.append(warning)
                    _print_rank0(rank, f"HEALTH WARNING: {warning}", flush=True)
                log_metrics["grad_norm"].append(loss.detach().new_tensor(float(grad_norm)))
                log_metrics["param_update_ratio"].append(loss.detach().new_tensor(float(update_ratio)))
                scheduler.step()   # cosine schedule advances every optimizer step
                optimizer.zero_grad()
                accum_count = 0
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    mean = _metric_means(log_metrics, distributed, world_size, device)
                    last_train_health = mean
                    health_warnings.extend(_health_warnings("pred", mean, args))
                    lr_now = optimizer.param_groups[0]["lr"]
                    # Curriculum weights at current step (for logging only)
                    log_patcs_w = _patcs_effective_weight(step, args.patcs_weight, args.patcs_warmup_steps)
                    log_prog_w = _progress_supervision_weight(
                        step, args.progress_supervision_weight, args.progress_supervision_decay_steps
                    )
                    record = {
                        "train/loss": mean["loss"],
                        "train/bc_loss": mean["bc_loss"],
                        "train/patcs_loss": mean["patcs_loss"],
                        "train/patcs_tube": mean["patcs_tube"],
                        "train/patcs_event": mean["patcs_event"],
                        "train/progress_loss": mean["progress_loss"],
                        "train/temporal_loss": mean["temporal_loss"],
                        "train/temporal_stall": mean["temporal_stall"],
                        "train/temporal_backward": mean["temporal_backward"],
                        "train/smoothness_loss": mean["smoothness_loss"],
                        "train/smooth_velocity": mean["smooth_velocity"],
                        "train/smooth_acceleration": mean["smooth_acceleration"],
                        "train/smooth_boundary": mean["smooth_boundary"],
                        "train/action_mean_abs": mean["action_mean_abs"],
                        "train/action_std": mean["action_std"],
                        "train/xyz_displacement": mean["xyz_displacement"],
                        "train/all_zero_rate": mean["all_zero_rate"],
                        "train/repeated_action_rate": mean["repeated_action_rate"],
                        "train/grad_norm": mean["grad_norm"],
                        "train/param_update_ratio": mean["param_update_ratio"],
                        "train/patcs_weight_effective": log_patcs_w,
                        "train/progress_supervision_weight_effective": log_prog_w,
                        "train/lr": lr_now,
                        "train/elapsed_sec_per_log": elapsed,
                        "train/per_device_batch": per_device_batch,
                        "train/grad_accum": grad_accum,
                        "train/world_size": world_size,
                        "train/effective_batch_size": per_device_batch * grad_accum * world_size,
                    }
                    if device.type == "cuda":
                        record.update(
                            {
                                "system/cuda_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
                                "system/cuda_reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
                                "system/cuda_max_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                            }
                        )
                    if rank == 0:
                        _log_swanlab(swanlab_run, record, step)
                    _print_rank0(
                        rank,
                        f"step={step:>6d}/{total_steps}  "
                        f"loss={mean['loss']:.4f}  bc={mean['bc_loss']:.4f}  "
                        f"patcs={mean['patcs_loss']:.4f}(w={log_patcs_w:.3f})  "
                        f"(tube={mean['patcs_tube']:.3f} event={mean['patcs_event']:.3f})  "
                        f"prog={mean['progress_loss']:.4f}(w={log_prog_w:.3f})  "
                        f"temp={mean['temporal_loss']:.4f} smooth={mean['smoothness_loss']:.4f}  "
                        f"|a|={mean['action_mean_abs']:.4f} xyz={mean['xyz_displacement']:.4f}  "
                        f"grad={mean['grad_norm']:.4f} upd={mean['param_update_ratio']:.2e}  "
                        f"zero={mean['all_zero_rate']:.3f} repeat={mean['repeated_action_rate']:.3f}  "
                        f"lr={lr_now:.2e}  {elapsed:.0f}s",
                        flush=True,
                    )
                    for v in log_metrics.values():
                        v.clear()
                    t0 = time.time()

                if rank == 0 and (step % args.checkpoint_every == 0 or step == total_steps):
                    checkpoint_path = _save_checkpoint(_unwrap_model(model), optimizer, scheduler, step, args, ckpt_dir)
                    if distributed or args.eval_async:
                        eval_path = _launch_checkpoint_eval_async(checkpoint_path, step, args, ckpt_dir)
                        if eval_path is not None:
                            _log_swanlab(
                                swanlab_run,
                                {
                                    "eval/launched": 1.0,
                                    "eval/async": 1.0,
                                    "eval/checkpoint_step": step,
                                },
                                step,
                            )
                    else:
                        eval_result = _run_checkpoint_eval(checkpoint_path, step, args, ckpt_dir)
                        if eval_result:
                            _log_swanlab(swanlab_run, _swanlab_eval_record(eval_result), step)
                    _rotate_checkpoints(ckpt_dir, keep=args.keep_last_checkpoints)

        # Final validation
        val_metrics = _run_validation(_unwrap_model(model), val_loader, mse_loss_fn, device) if rank == 0 else None
        if rank == 0 and val_metrics is not None:
            print(
                f"Validation: bc_loss={val_metrics['bc_loss']:.4f}  "
                f"({val_metrics['num_samples']} samples)",
                flush=True,
            )
            _log_swanlab(
                swanlab_run,
                {
                    "val/bc_loss": val_metrics["bc_loss"],
                    "val/num_samples": val_metrics["num_samples"],
                },
                step,
            )

            # Write final summary
            summary = {
                "final_step": step,
                "val_bc_loss": val_metrics["bc_loss"],
                "world_size": world_size,
                "data_health": data_health,
                "last_train_health": last_train_health,
                "health_warnings": sorted(set(health_warnings)),
                "args": _json_safe(vars(args)),
            }
            (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"Training complete. Checkpoints in {ckpt_dir}", flush=True)
    finally:
        if rank == 0:
            _finish_swanlab(swanlab_run)
        _cleanup_distributed(distributed)


def _run_validation(
    model: ActionMLPPolicy,
    val_loader: DataLoader,
    loss_fn: nn.MSELoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            obs = batch["obs"].to(device, dtype=torch.float32)
            action_gt = batch["action_chunk"].to(device, dtype=torch.float32)
            pred = model(obs)
            total_loss += loss_fn(pred, action_gt).item() * obs.shape[0]
            n += obs.shape[0]
    model.train()
    return {"bc_loss": total_loss / max(n, 1), "num_samples": n}


def _libero_benchmark_name(suite: str) -> str:
    return suite if suite.startswith("libero_") else f"libero_{suite}"


def _checkpoint_eval_payload(
    checkpoint_path: Path,
    step: int,
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> tuple[dict, Path]:
    output_dir = args.eval_output_dir or (ckpt_dir / "libero_eval" / f"step_{step:07d}")
    payload = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "benchmark": args.eval_benchmark or _libero_benchmark_name(args.suite),
        "task_id": args.eval_task_id,
        "trials": args.eval_trials,
        "max_steps": args.eval_max_steps,
        "settle_steps": args.eval_settle_steps,
        "camera_size": args.eval_camera_size,
        "camera_name": args.eval_camera_name,
        "fps": args.eval_fps,
        "seed": args.seed,
        "save_video": args.eval_save_video,
        "cpu": args.eval_cpu,
        "libero_root": str(args.libero_root) if args.libero_root is not None else None,
    }
    return payload, Path(output_dir)


def _checkpoint_eval_command(payload: dict) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "new_il.libero.evaluate_checkpoint",
        "--checkpoint",
        payload["checkpoint"],
        "--output-dir",
        payload["output_dir"],
        "--benchmark",
        payload["benchmark"],
        "--task-id",
        str(payload["task_id"]),
        "--trials",
        str(payload["trials"]),
        "--max-steps",
        str(payload["max_steps"]),
        "--settle-steps",
        str(payload["settle_steps"]),
        "--camera-size",
        str(payload["camera_size"]),
        "--camera-name",
        payload["camera_name"],
        "--fps",
        str(payload["fps"]),
        "--seed",
        str(payload["seed"]),
    ]
    if payload["save_video"]:
        command.append("--save-video")
    if payload["cpu"]:
        command.append("--cpu")
    if payload["libero_root"] is not None:
        command.extend(["--libero-root", str(payload["libero_root"])])
    return command


def _run_checkpoint_eval(
    checkpoint_path: Path,
    step: int,
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> dict | None:
    if not args.eval_on_checkpoint:
        return None

    payload, output_dir = _checkpoint_eval_payload(checkpoint_path, step, args, ckpt_dir)
    if args.eval_server_url:
        result = _run_checkpoint_eval_client(args.eval_server_url, payload, output_dir, args.eval_timeout_sec)
        if result:
            print(f"LIBERO eval step={step}: {output_dir / 'result.json'}", flush=True)
            return result

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            _checkpoint_eval_command(payload),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        failure = {
            "status": "failed",
            "return_code": result.returncode,
            "checkpoint": str(checkpoint_path),
            "log": str(log_path),
        }
        (output_dir / "result.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        result_payload = failure
    else:
        result_path = output_dir / "result.json"
        if result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result_payload = {"status": "missing_result", "checkpoint": str(checkpoint_path)}
    print(f"LIBERO eval step={step}: {output_dir / 'result.json'}", flush=True)
    return result_payload


def _launch_checkpoint_eval_async(
    checkpoint_path: Path,
    step: int,
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> Path | None:
    if not args.eval_on_checkpoint:
        return None

    payload, output_dir = _checkpoint_eval_payload(checkpoint_path, step, args, ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / checkpoint_path.name
    if not snapshot_path.exists():
        try:
            os.link(checkpoint_path, snapshot_path)
        except OSError:
            shutil.copy2(checkpoint_path, snapshot_path)
    payload["checkpoint"] = str(snapshot_path)
    request_path = output_dir / "request.json"
    log_path = output_dir / "eval_async.log"
    request_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.eval_server_url:
        command = [
            sys.executable,
            "-m",
            "new_il.libero.eval_request",
            "--server-url",
            args.eval_server_url,
            "--payload",
            str(request_path),
            "--timeout-sec",
            str(args.eval_timeout_sec),
        ]
    else:
        command = _checkpoint_eval_command(payload)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.Popen(  # noqa: S603
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
    print(f"LIBERO eval step={step} launched asynchronously: {output_dir}", flush=True)
    return output_dir


def _swanlab_eval_record(result: dict) -> dict:
    record = {
        "eval/status_completed": 1.0 if result.get("status") == "completed" else 0.0,
        "eval/status_failed": 1.0 if result.get("status") == "failed" else 0.0,
        "eval/status_skipped": 1.0 if result.get("status") == "skipped" else 0.0,
    }
    for key in ("success_rate", "successes", "trials", "elapsed_sec"):
        if key in result and isinstance(result[key], (int, float)):
            record[f"eval/{key}"] = result[key]
    if "total_trials" in result and isinstance(result["total_trials"], (int, float)):
        record["eval/total_trials"] = result["total_trials"]
    task_results = result.get("task_results")
    if isinstance(task_results, list):
        record["eval/task_count"] = len(task_results)
    videos = result.get("videos")
    if isinstance(videos, list):
        record["eval/video_count"] = len(videos)
    return record


def _run_checkpoint_eval_client(
    server_url: str,
    payload: dict,
    output_dir: Path,
    timeout_sec: float,
) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    request = Request(
        server_url.rstrip("/") + "/eval",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError) as exc:
        failure = {
            "status": "failed",
            "reason": f"LIBERO eval server request failed: {type(exc).__name__}: {exc}",
            "server_url": server_url,
            "fallback": "subprocess",
        }
        (output_dir / "server_error.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        return None
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _save_checkpoint(
    model: ActionMLPPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> Path:
    path = ckpt_dir / f"ckpt_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
        },
        path,
    )
    return path


def _rotate_checkpoints(ckpt_dir: Path, keep: int) -> None:
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LIBERO action-chunk policy (BC + PA-TCS).")

    # Data
    parser.add_argument("--suite", default="object", help="Suite name (for logging).")
    parser.add_argument(
        "--hdf5-root", type=Path, default=None,
        help="Directory with per-task .hdf5 files (e.g. data/libero_rlds_hdf5/object).",
    )
    parser.add_argument(
        "--rollout-root", type=Path, default=None,
        help="Cleaned rollout by_task root or one task dir from new-il-rollout-manifest.",
    )
    parser.add_argument(
        "--artifact-root", type=Path, required=True,
        help="Directory with matching *_patcs.npz files.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Cap number of tasks (smoke).")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--obs-full-state", action="store_true",
                        help="Use ee_states [8] instead of ee_pos [3] as obs.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. Default 0: data is in-memory, workers add overhead.")
    parser.add_argument("--seed", type=int, default=7)

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--per-device-batch-size", type=int, default=32)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--patcs-weight", type=float, default=0.1,
                        help="Set to 0.0 for pure BC baseline.")
    parser.add_argument("--event-weight", type=float, default=10.0)
    parser.add_argument(
        "--event-clip", type=float, default=25.0,
        help="Clip each normalized event loss term before averaging. <=0 disables clipping.",
    )
    # Progress curriculum — addresses 'stay-in-place / open-gripper' local optimum
    parser.add_argument(
        "--patcs-warmup-steps", type=int, default=5000,
        help="Ramp PATCS weight from 0 → patcs-weight over this many optimizer steps. "
             "Keeps noisy early-training DTW supervision from dominating BC. 0 = no warmup.",
    )
    parser.add_argument(
        "--progress-supervision-weight", type=float, default=0.5,
        help="Initial weight of the direct progress-forward loss (penalizes zero displacement). "
             "Decays to 0 over progress-supervision-decay-steps. 0 = disabled.",
    )
    parser.add_argument(
        "--progress-supervision-decay-steps", type=int, default=8000,
        help="Steps over which the progress-forward loss decays from its initial weight to 0.",
    )
    parser.add_argument(
        "--progress-min-displacement", type=float, default=0.02,
        help="Minimum expected net end-effector displacement (metres) per action chunk. "
             "Chunks with smaller displacement incur a progress-forward penalty.",
    )
    parser.add_argument(
        "--temporal-consistency-weight", type=float, default=0.0,
        help="Weight for static stall/backward progress proxy loss.",
    )
    parser.add_argument(
        "--temporal-min-step-displacement", type=float, default=0.0025,
        help="Minimum per-step xyz action norm for stall proxy loss.",
    )
    parser.add_argument(
        "--smoothness-weight", type=float, default=0.0,
        help="Weight for intra-chunk and boundary smoothness loss.",
    )
    parser.add_argument("--smooth-velocity-weight", type=float, default=1.0)
    parser.add_argument("--smooth-acceleration-weight", type=float, default=0.25)
    parser.add_argument("--smooth-boundary-weight", type=float, default=0.25)
    parser.add_argument(
        "--dynamic-trajectory",
        action="store_true",
        help="Use simplified dynamic trajectory-worker training. Requires --rollout-root.",
    )
    parser.add_argument(
        "--dynamic-workers",
        type=int,
        default=None,
        help="Number of dynamic worker states. Defaults to per-device batch size.",
    )
    parser.add_argument(
        "--dynamic-min-advance",
        type=int,
        default=1,
        help="Minimum demo-index advance after matching a predicted chunk.",
    )
    parser.add_argument(
        "--dynamic-max-advance",
        type=int,
        default=None,
        help="Maximum demo-index advance after matching. Defaults to action horizon.",
    )
    parser.add_argument(
        "--dynamic-matcher",
        choices=["artifact_anchor", "artifact_anchor_path_mse", "artifact_cloud_path_mse", "demo_nearest"],
        default="artifact_anchor",
        help="Progress matcher used by simplified dynamic trajectory workers.",
    )
    parser.add_argument(
        "--dynamic-match-cost",
        choices=["mse", "l2"],
        default="mse",
        help="Point distance cost for terminal-point dynamic matchers.",
    )
    parser.add_argument("--health-min-xyz-norm", type=float, default=1e-5)
    parser.add_argument("--health-max-all-zero-rate", type=float, default=0.05)
    parser.add_argument("--health-max-repeated-action-rate", type=float, default=0.25)
    parser.add_argument("--health-max-stall-rate", type=float, default=0.5)
    parser.add_argument("--health-max-update-ratio", type=float, default=0.2)
    parser.add_argument(
        "--lr-warmup-steps", type=int, default=1000,
        help="Steps for linear LR warmup (0 → lr). Cosine decay runs from warmup end to max-steps. "
             "Coordinate with --patcs-warmup-steps so LR is at full strength before PATCS activates.",
    )
    parser.add_argument("--min-lr", type=float, default=1e-7,
                        help="Cosine decay floor. Final LR approaches this value at max-steps.")

    # PATCS tube config
    parser.add_argument("--tube-sigma", type=float, default=0.05)
    parser.add_argument("--tube-temperature", type=float, default=0.25)
    parser.add_argument("--tube-v-max", type=float, default=0.2)
    parser.add_argument("--tube-delta", type=float, default=0.08)

    # Checkpointing / logging
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/checkpoints/default"))
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--keep-last-checkpoints", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-on-checkpoint", action="store_true",
                        help="Run LIBERO evaluation every time a checkpoint is saved.")
    parser.add_argument("--eval-output-dir", type=Path, default=None,
                        help="Override checkpoint eval output directory.")
    parser.add_argument("--eval-benchmark", default=None,
                        help="LIBERO benchmark name. Defaults to libero_<suite>.")
    parser.add_argument("--eval-task-id", default="0",
                        help="LIBERO task id, or 'all' to evaluate every task in the benchmark.")
    parser.add_argument("--eval-trials", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=400)
    parser.add_argument("--eval-settle-steps", type=int, default=5)
    parser.add_argument("--eval-camera-size", type=int, default=128)
    parser.add_argument("--eval-camera-name", default="agentview_image")
    parser.add_argument("--eval-fps", type=int, default=20)
    parser.add_argument("--eval-save-video", action="store_true")
    parser.add_argument("--eval-cpu", action="store_true")
    parser.add_argument("--eval-server-url", default=os.environ.get("NEW_IL_LIBERO_EVAL_SERVER"),
                        help="Optional LIBERO eval server URL, e.g. http://127.0.0.1:8765.")
    parser.add_argument("--eval-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--eval-async", action="store_true",
                        help="Launch checkpoint evaluation in a detached process. Always enabled for DDP.")
    parser.add_argument("--libero-root", type=Path, default=os.environ.get("LIBERO_ROOT"))

    # Live visualization
    parser.add_argument("--swanlab", action="store_true",
                        help="Log live training curves to SwanLab on rank 0.")
    parser.add_argument("--swanlab-project", default="new-il")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-mode", choices=["cloud", "local", "offline", "disabled"], default="cloud")
    parser.add_argument("--swanlab-logdir", type=Path, default=None)

    args = parser.parse_args()
    if args.swanlab_mode == "disabled":
        args.swanlab = False
    train(args)


if __name__ == "__main__":
    main()
