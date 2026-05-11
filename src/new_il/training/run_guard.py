from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from new_il.training.memory import GpuInfo, plan_memory


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _query_gpus() -> list[dict[str, int]]:
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
        gpus.append({"index": index, "total_mb": total, "free_mb": free})
    return gpus


def _select_gpus(gpus: list[dict[str, int]], min_free_gb: float, max_gpus: int) -> list[int]:
    eligible = [gpu for gpu in gpus if gpu["free_mb"] >= min_free_gb * 1024]
    eligible.sort(key=lambda gpu: gpu["free_mb"], reverse=True)
    return [gpu["index"] for gpu in eligible[:max_gpus]]


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("pyyaml is required to load config files.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _is_python_executable(value: str) -> bool:
    name = Path(value).name
    return name == "python" or name.startswith("python")


def _torchrun_command(command: list[str], nproc_per_node: int) -> list[str]:
    """Translate common Python module commands into torchrun form."""
    base = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={nproc_per_node}",
    ]
    if len(command) >= 3 and _is_python_executable(command[0]) and command[1] == "-m":
        return [*base, "--module", command[2], *command[3:]]
    return [*base, *command]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a training command with GPU selection and durable logs."
    )
    parser.add_argument("--name", default="train")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/new_il"))
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--target-global-batch", type=int, default=128)
    parser.add_argument("--reserve-gb", type=float, default=2.0)
    parser.add_argument("--gb-per-sample", type=float, default=0.75)
    parser.add_argument("--batch-multiple", type=int, default=4)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--allow-oom-risk", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Pass a command after '--', for example: new-il-run -- python train.py")

    run_dir = args.runs_dir / args.name / f"{_timestamp()}_{args.name}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)

    gpus = _query_gpus()
    memory_plan = plan_memory(
        [
            GpuInfo(index=gpu["index"], total_mb=gpu["total_mb"], free_mb=gpu["free_mb"])
            for gpu in gpus
        ],
        min_free_gb=args.min_free_gb,
        max_gpus=args.max_gpus,
        target_global_batch=args.target_global_batch,
        memory_fraction=args.memory_fraction,
        reserve_gb=args.reserve_gb,
        gb_per_sample=args.gb_per_sample,
        batch_multiple=args.batch_multiple,
        per_device_batch_size=args.per_device_batch_size,
        allow_oom_risk=args.allow_oom_risk,
    )
    selected = memory_plan.selected_gpus or _select_gpus(gpus, args.min_free_gb, args.max_gpus)
    launch_command = _torchrun_command(command, len(selected)) if len(selected) > 1 else command

    env = os.environ.copy()
    if selected:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in selected)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.memory_fraction))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("NEW_IL_PER_DEVICE_BATCH_SIZE", str(memory_plan.per_device_batch_size))
    env.setdefault("NEW_IL_GRAD_ACCUMULATION_STEPS", str(memory_plan.grad_accumulation_steps))
    env.setdefault("NEW_IL_EFFECTIVE_BATCH_SIZE", str(memory_plan.effective_batch_size))

    run_config = {
        "name": args.name,
        "config_path": str(args.config) if args.config else None,
        "config": _load_config(args.config),
        "command": command,
        "launch_command": launch_command,
        "run_dir": str(run_dir),
        "gpu_inventory": gpus,
        "selected_gpus": selected,
        "memory_plan": {
            "selected_gpus": memory_plan.selected_gpus,
            "per_device_batch_size": memory_plan.per_device_batch_size,
            "grad_accumulation_steps": memory_plan.grad_accumulation_steps,
            "effective_batch_size": memory_plan.effective_batch_size,
            "max_safe_per_device_batch_size": memory_plan.max_safe_per_device_batch_size,
            "requested_per_device_batch_size": memory_plan.requested_per_device_batch_size,
            "oom_risk": memory_plan.oom_risk,
            "batch_multiple": memory_plan.batch_multiple,
            "weakest_free_gb": memory_plan.weakest_free_gb,
            "reason": memory_plan.reason,
        },
        "min_free_gb": args.min_free_gb,
        "memory_fraction": args.memory_fraction,
        "reserve_gb": args.reserve_gb,
        "gb_per_sample": args.gb_per_sample,
        "batch_multiple": args.batch_multiple,
        "environment_overrides": {
            key: env[key]
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "TOKENIZERS_PARALLELISM",
                "PYTHONUNBUFFERED",
                "NEW_IL_PER_DEVICE_BATCH_SIZE",
                "NEW_IL_GRAD_ACCUMULATION_STEPS",
                "NEW_IL_EFFECTIVE_BATCH_SIZE",
            ]
            if key in env
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    if args.dry_run:
        (run_dir / "status.json").write_text(json.dumps({"status": "dry_run"}, indent=2) + "\n")
        print(run_dir)
        return

    log_path = logs_dir / "terminal.log"
    status_path = run_dir / "status.json"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"event": "start", "command": command}) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            launch_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    status = {
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "terminal_log": str(log_path),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    raise SystemExit(return_code)
