from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    total_mb: int
    free_mb: int


@dataclass(frozen=True)
class MemoryPlan:
    selected_gpus: list[int]
    per_device_batch_size: int
    grad_accumulation_steps: int
    effective_batch_size: int
    max_safe_per_device_batch_size: int
    requested_per_device_batch_size: int | None
    oom_risk: bool
    batch_multiple: int
    weakest_free_gb: float | None
    reason: str


def query_gpus() -> list[GpuInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, total, free = map(int, parts)
        gpus.append(GpuInfo(index=index, total_mb=total, free_mb=free))
    return gpus


def _round_down_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("batch_multiple must be positive.")
    return max(0, (value // multiple) * multiple)


def max_batch_from_free_memory(
    free_gb: float,
    *,
    memory_fraction: float = 0.90,
    reserve_gb: float = 2.0,
    gb_per_sample: float = 0.75,
    batch_multiple: int = 4,
) -> int:
    """Estimate the largest safe per-device batch, rounded to a multiple."""

    usable_gb = free_gb * memory_fraction - reserve_gb
    if usable_gb <= 0:
        return 0
    raw = int(usable_gb // max(gb_per_sample, 1e-8))
    return _round_down_multiple(raw, batch_multiple)


def plan_memory(
    gpus: list[GpuInfo],
    *,
    min_free_gb: float = 20.0,
    max_gpus: int = 1,
    target_global_batch: int = 128,
    memory_fraction: float = 0.90,
    reserve_gb: float = 2.0,
    gb_per_sample: float = 0.75,
    batch_multiple: int = 4,
    per_device_batch_size: int | None = None,
    allow_oom_risk: bool = False,
    min_per_device_batch: int | None = None,
    max_per_device_batch: int | None = None,
) -> MemoryPlan:
    eligible = [gpu for gpu in gpus if gpu.free_mb >= min_free_gb * 1024]
    eligible.sort(key=lambda gpu: gpu.free_mb, reverse=True)
    selected = eligible[:max_gpus]
    if not selected:
        return MemoryPlan(
            selected_gpus=[],
            per_device_batch_size=1,
            grad_accumulation_steps=max(1, target_global_batch),
            effective_batch_size=max(1, target_global_batch),
            max_safe_per_device_batch_size=1,
            requested_per_device_batch_size=per_device_batch_size,
            oom_risk=False,
            batch_multiple=batch_multiple,
            weakest_free_gb=None,
            reason="no eligible GPU found; use CPU/debug defaults",
        )

    weakest_free_gb = min(gpu.free_mb for gpu in selected) / 1024.0
    max_safe = max_batch_from_free_memory(
        weakest_free_gb,
        memory_fraction=memory_fraction,
        reserve_gb=reserve_gb,
        gb_per_sample=gb_per_sample,
        batch_multiple=batch_multiple,
    )
    if max_safe < batch_multiple:
        max_safe = 1
    if per_device_batch_size is not None:
        if per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive.")
        if per_device_batch_size % batch_multiple != 0:
            raise ValueError(
                f"manual per-device batch {per_device_batch_size} must be a multiple of {batch_multiple}."
            )
        oom_risk = per_device_batch_size > max_safe
        if oom_risk and not allow_oom_risk:
            raise ValueError(
                "manual per-device batch "
                f"{per_device_batch_size} exceeds safe estimate {max_safe} on the weakest selected GPU "
                f"({weakest_free_gb:.1f} GiB free). Re-run with --allow-oom-risk to override."
            )
        per_device = per_device_batch_size
        reason = (
            f"manual per-device batch {per_device}; safe estimate {max_safe} from weakest selected GPU "
            f"free memory: {weakest_free_gb:.1f} GiB"
        )
    else:
        per_device = max_safe
        # Apply explicit upper/lower clamps in auto mode.
        if max_per_device_batch is not None and per_device > max_per_device_batch:
            per_device = _round_down_multiple(max_per_device_batch, batch_multiple)
        if min_per_device_batch is not None and per_device < min_per_device_batch:
            per_device = min_per_device_batch
        if per_device < 1:
            per_device = 1
        oom_risk = False
        reason = (
            f"auto per-device batch {per_device}, rounded to multiple of {batch_multiple}, "
            f"from weakest selected GPU free memory: {weakest_free_gb:.1f} GiB"
            + (f"; capped at max_per_device_batch={max_per_device_batch}" if max_per_device_batch and per_device == _round_down_multiple(max_per_device_batch, batch_multiple) else "")
            + (f"; floored at min_per_device_batch={min_per_device_batch}" if min_per_device_batch and per_device == min_per_device_batch else "")
        )
    physical_batch = per_device * len(selected)
    accumulation = max(1, math.ceil(target_global_batch / physical_batch))
    return MemoryPlan(
        selected_gpus=[gpu.index for gpu in selected],
        per_device_batch_size=per_device,
        grad_accumulation_steps=accumulation,
        effective_batch_size=physical_batch * accumulation,
        max_safe_per_device_batch_size=max_safe,
        requested_per_device_batch_size=per_device_batch_size,
        oom_risk=oom_risk,
        batch_multiple=batch_multiple,
        weakest_free_gb=weakest_free_gb,
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan GPU batch and accumulation settings.")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--target-global-batch", type=int, default=128)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--reserve-gb", type=float, default=2.0)
    parser.add_argument("--gb-per-sample", type=float, default=0.75)
    parser.add_argument("--batch-multiple", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--allow-oom-risk", action="store_true")
    parser.add_argument("--min-per-device-batch", type=int, default=None,
                        help="Auto-mode lower clamp for per-device batch size.")
    parser.add_argument("--max-per-device-batch", type=int, default=None,
                        help="Auto-mode upper clamp for per-device batch size.")
    args = parser.parse_args()
    plan = plan_memory(
        query_gpus(),
        min_free_gb=args.min_free_gb,
        max_gpus=args.max_gpus,
        target_global_batch=args.target_global_batch,
        memory_fraction=args.memory_fraction,
        reserve_gb=args.reserve_gb,
        gb_per_sample=args.gb_per_sample,
        batch_multiple=args.batch_multiple,
        per_device_batch_size=args.per_device_batch_size,
        allow_oom_risk=args.allow_oom_risk,
        min_per_device_batch=args.min_per_device_batch,
        max_per_device_batch=args.max_per_device_batch,
    )
    print(json.dumps(asdict(plan), indent=2))
