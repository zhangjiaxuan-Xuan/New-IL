from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np


CLAIM_DONE = "done"
CLAIM_WAIT = "wait"
CLAIM_JOB = "job"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _count_dir(path: Path) -> int:
    return sum(1 for _ in path.iterdir()) if path.exists() else 0


def _task_key(task_suite_name: str, task_id: int) -> str:
    return f"{task_suite_name}_task_{task_id:02d}"


def _task_dir_name(task_suite_name: str, task_id: int) -> str:
    return _task_key(task_suite_name, task_id)


def _job_task_key(job: dict[str, Any]) -> str:
    return _task_key(str(job["task_suite_name"]), int(job["task_id"]))


def _ledger_dirs(queue_dir: Path) -> tuple[Path, Path]:
    ledger = queue_dir / "ledger"
    return ledger / "done", ledger / "success"


def record_episode_done(queue_dir: Path, worker_id: int) -> None:
    done_dir, _ = _ledger_dirs(queue_dir)
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / f"w{worker_id:02d}_{uuid.uuid4().hex}").write_text("", encoding="utf-8")


def record_success(
    queue_dir: Path,
    task_id: int,
    worker_id: int,
    task_suite_name: str = "libero_spatial",
) -> None:
    _, success_root = _ledger_dirs(queue_dir)
    task_dir = success_root / _task_dir_name(task_suite_name, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"w{worker_id:02d}_{uuid.uuid4().hex}").write_text("", encoding="utf-8")


def success_counts(queue_dir: Path) -> dict[str, int]:
    _, success_root = _ledger_dirs(queue_dir)
    counts: dict[str, int] = {}
    if not success_root.exists():
        return counts
    for path in success_root.glob("task_*"):
        counts[path.name] = _count_dir(path)
    for path in success_root.glob("libero_*_task_*"):
        counts[path.name] = _count_dir(path)
    return counts


def inflight_counts(queue_dir: Path) -> dict[str, int]:
    running = queue_dir / "running"
    counts: dict[str, int] = {}
    if not running.exists():
        return counts
    for path in running.glob("*.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            key = _job_task_key(job)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def collection_progress(queue_dir: Path) -> tuple[int, dict[str, int]]:
    done_dir, _ = _ledger_dirs(queue_dir)
    return _count_dir(done_dir), success_counts(queue_dir)


def claim_job_deficit(
    queue_dir: Path,
    worker_id: int,
    per_task_target: int,
    total_target: int,
) -> tuple[str, dict[str, Any] | None, Path | None]:
    """Claim a queued LIBERO job with Mem-style in-flight deficit scheduling."""

    queue_dir = Path(queue_dir)
    running = queue_dir / "running"
    running.mkdir(parents=True, exist_ok=True)
    success = success_counts(queue_dir)
    if sum(success.values()) >= total_target:
        return CLAIM_DONE, None, None

    pending_root = queue_dir / "pending"
    task_dirs: dict[str, Path] = {}
    if pending_root.exists():
        for path in pending_root.iterdir():
            if path.is_dir():
                task_dirs[path.name] = path
    if not task_dirs:
        return CLAIM_DONE, None, None

    inflight = inflight_counts(queue_dir)
    pool_open = {task_id: next(task_dir.glob("*.json"), None) is not None for task_id, task_dir in task_dirs.items()}
    balanced_ceiling = sum(
        per_task_target if pool_open[task_id] else min(per_task_target, success.get(task_id, 0))
        for task_id in task_dirs
    )
    cap = 10**9 if balanced_ceiling < total_target else per_task_target

    candidates: list[tuple[int, str, Path]] = []
    blocked_by_inflight = False
    for task_key, task_dir in task_dirs.items():
        if not pool_open[task_key]:
            continue
        effective = success.get(task_key, 0) + inflight.get(task_key, 0)
        if effective < cap:
            candidates.append((effective, task_key, task_dir))
        elif success.get(task_key, 0) < cap:
            blocked_by_inflight = True
    candidates.sort(key=lambda item: (item[0], item[1]))

    for _, _, task_dir in candidates:
        for path in sorted(task_dir.glob("*.json")):
            claimed = running / f"{task_dir.name}_{path.stem}.w{worker_id:02d}.json"
            try:
                os.replace(path, claimed)
            except OSError:
                continue
            return CLAIM_JOB, json.loads(claimed.read_text(encoding="utf-8")), claimed

    if blocked_by_inflight or sum(inflight.values()) > 0:
        return CLAIM_WAIT, None, None
    return CLAIM_DONE, None, None


def make_openpi_queue(
    queue_dir: Path,
    task_suite_names: list[str] | str | None = None,
    task_ids: list[int] | None = None,
    attempts_per_task: int = 50,
    seed: int = 7,
    *,
    task_suite_name: str | None = None,
) -> int:
    if task_suite_names is None:
        task_suite_names = task_suite_name or "libero_spatial"
    if isinstance(task_suite_names, str):
        task_suite_names = [task_suite_names]
    if task_ids is None:
        task_ids = [0]

    pending = queue_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    n_jobs = 0
    for task_suite_name in task_suite_names:
        for task_id in task_ids:
            task_dir = pending / _task_dir_name(task_suite_name, task_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            for episode_idx in range(attempts_per_task):
                job = {
                    "task_suite_name": task_suite_name,
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "seed": seed + episode_idx,
                }
                _write_json(task_dir / f"job_{episode_idx:05d}.json", job)
                n_jobs += 1
    _write_json(
        queue_dir / "queue_meta.json",
        {
            "task_suite_names": task_suite_names,
            "task_ids": task_ids,
            "attempts_per_task": attempts_per_task,
            "seed": seed,
            "jobs": n_jobs,
        },
    )
    return n_jobs


def requeue_running_jobs(queue_dir: Path) -> int:
    queue_dir = Path(queue_dir)
    running = queue_dir / "running"
    pending = queue_dir / "pending"
    if not running.exists():
        return 0

    restored = 0
    for claimed_path in sorted(running.glob("*.json")):
        try:
            job = json.loads(claimed_path.read_text(encoding="utf-8"))
            task_dir = pending / _job_task_key(job)
            task_dir.mkdir(parents=True, exist_ok=True)
            pending_path = task_dir / f"job_{int(job['episode_idx']):05d}.json"
            if pending_path.exists():
                claimed_path.unlink(missing_ok=True)
                continue
            os.replace(claimed_path, pending_path)
            restored += 1
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return restored


def _parse_task_ids(text: str) -> list[int]:
    if "," in text:
        return [int(item) for item in text.split(",") if item.strip()]
    if "-" in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(text)]


def _parse_task_suites(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def make_queue_main() -> None:
    parser = argparse.ArgumentParser(description="Create a New-IL OpenPI LIBERO queue.")
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--task-suite-name", default=None)
    parser.add_argument(
        "--task-suite-names",
        default="libero_spatial",
        help="Comma list such as libero_spatial,libero_object,libero_goal,libero_10.",
    )
    parser.add_argument("--task-ids", default="0", help="Single id, comma list, or inclusive range such as 0-9.")
    parser.add_argument("--attempts-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    n_jobs = make_openpi_queue(
        args.queue_dir,
        _parse_task_suites(args.task_suite_name or args.task_suite_names),
        _parse_task_ids(args.task_ids),
        args.attempts_per_task,
        args.seed,
    )
    print(json.dumps({"status": "ok", "queue_dir": str(args.queue_dir), "jobs": n_jobs}, indent=2))


def requeue_running_main() -> None:
    parser = argparse.ArgumentParser(description="Move stale OpenPI LIBERO running jobs back to pending.")
    parser.add_argument("--queue-dir", type=Path, required=True)
    args = parser.parse_args()
    restored = requeue_running_jobs(args.queue_dir)
    print(json.dumps({"status": "ok", "queue_dir": str(args.queue_dir), "restored": restored}, indent=2))


def worker_main() -> None:
    parser = argparse.ArgumentParser(description="Run a New-IL OpenPI LIBERO queue worker.")
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--per-task-target", type=int, default=10)
    parser.add_argument("--total-target", type=int, default=40)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--libero-root", type=Path, default=Path("third_party/LIBERO"))
    parser.add_argument("--poll-sec", type=float, default=2.0)
    args = parser.parse_args()

    import time
    from new_il.libero.rollout import LiberoRolloutConfig, OpenPIChunkPolicy, prepare_libero_paths, run_one_episode

    prepare_libero_paths(args.libero_root, args.output_dir)
    from libero.libero.benchmark import get_benchmark_dict

    benchmark_cache: dict[str, Any] = {}
    policy = OpenPIChunkPolicy(
        host=args.host,
        port=args.port,
        replan_steps=args.replan_steps,
        resize_size=args.resize_size,
    )
    started = time.time()
    episodes = 0
    successes = 0
    actions_all: list[np.ndarray] = []
    while True:
        status, job, claimed_path = claim_job_deficit(
            args.queue_dir,
            args.worker_id,
            args.per_task_target,
            args.total_target,
        )
        if status == CLAIM_DONE:
            break
        if status == CLAIM_WAIT:
            time.sleep(args.poll_sec)
            continue
        assert job is not None and claimed_path is not None
        task_output = args.output_dir / f"worker_{args.worker_id:02d}" / f"task_{int(job['task_id']):02d}_ep_{int(job['episode_idx']):05d}"
        suite_name = str(job["task_suite_name"])
        if suite_name not in benchmark_cache:
            benchmark_cache[suite_name] = get_benchmark_dict()[suite_name]()
        task = benchmark_cache[suite_name].get_task(int(job["task_id"]))
        rollout_config = LiberoRolloutConfig(
            task_suite_name=job["task_suite_name"],
            max_steps=args.max_steps or _max_steps(str(job["task_suite_name"])),
            settle_steps=args.settle_steps,
            resolution=args.camera_size,
            fps=args.fps,
            save_video=True,
            save_traj=True,
        )
        success = False
        try:
            ep_result, traj_path, actions = run_one_episode(
                config=rollout_config,
                task=task,
                language=str(getattr(task, "language", "")),
                episode_idx=int(job["episode_idx"]),
                seed=int(job["seed"]),
                policy=policy,
                output_dir=task_output,
                task_idx=int(job["task_id"]),
            )
            success = bool(ep_result.success)
            episodes += 1
            successes += int(success)
            if actions is not None:
                actions_all.append(actions)
            record_episode_done(args.queue_dir, args.worker_id)
            if success:
                record_success(
                    args.queue_dir,
                    task_id=int(job["task_id"]),
                    worker_id=args.worker_id,
                    task_suite_name=str(job["task_suite_name"]),
                )
        finally:
            Path(claimed_path).unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "event": "openpi_queue_worker_job",
                    "worker_id": args.worker_id,
                    "task_suite_name": str(job["task_suite_name"]),
                    "task_id": int(job["task_id"]),
                    "episode_idx": int(job["episode_idx"]),
                    "success": success,
                    "traj_npz_path": str(traj_path) if traj_path else None,
                    "episodes": episodes,
                    "successes": successes,
                }
            ),
            flush=True,
        )

    done, counts = collection_progress(args.queue_dir)
    summary = {
        "status": "done",
        "worker_id": args.worker_id,
        "episodes": episodes,
        "successes": successes,
        "queue_done": done,
        "queue_success_counts": counts,
        "action_count": int(sum(len(a) for a in actions_all)),
        "elapsed_sec": time.time() - started,
    }
    _write_json(args.output_dir / f"worker_{args.worker_id:02d}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def _max_steps(task_suite_name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(task_suite_name, 400)
