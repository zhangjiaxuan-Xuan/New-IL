from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is required. Run: uv sync --extra train") from exc

from new_il.training.model import ActionMLPPolicy


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _import_libero_api():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def _prepare_libero_paths(libero_root: Path | None, output_dir: Path) -> None:
    if libero_root is None:
        return
    libero_root = libero_root.expanduser().resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))

    config_dir = output_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    if not config_path.exists():
        package_root = libero_root / "libero"
        benchmark_root = package_root / "libero"
        config_path.write_text(
            "\n".join(
                [
                    f"benchmark_root: {benchmark_root}",
                    f"bddl_files: {benchmark_root / 'bddl_files'}",
                    f"init_states: {benchmark_root / 'init_files'}",
                    f"datasets: {package_root / 'datasets'}",
                    f"assets: {benchmark_root / 'assets'}",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _benchmark_names(name: str) -> list[str]:
    if name == "all":
        return ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    return [name]


def _task_ids(task_suite, task_id: str) -> list[int]:
    if str(task_id).lower() == "all":
        return list(range(task_suite.n_tasks))
    return [int(task_id)]


def _load_model(checkpoint: Path, device: torch.device) -> tuple[ActionMLPPolicy, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device)
    args = payload.get("args", {})
    model = ActionMLPPolicy(
        obs_dim=8 if args.get("obs_full_state") else 3,
        action_dim=7,
        horizon=int(args.get("action_horizon", 8)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        num_layers=int(args.get("num_layers", 3)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, args


def _obs_vector(obs: dict[str, Any], obs_full_state: bool) -> np.ndarray:
    ee_pos = None
    for key in ("ee_pos", "robot0_eef_pos", "eef_pos"):
        if key in obs:
            ee_pos = np.asarray(obs[key], dtype=np.float32).reshape(-1)[:3]
            break
    if ee_pos is None:
        raise KeyError(f"Could not find end-effector position in obs keys: {sorted(obs.keys())}")
    if not obs_full_state:
        return ee_pos

    candidates = [ee_pos]
    for key in ("robot0_eef_quat", "eef_quat", "ee_quat"):
        if key in obs:
            candidates.append(np.asarray(obs[key], dtype=np.float32).reshape(-1)[:4])
            break
    for key in ("robot0_gripper_qpos", "gripper_qpos"):
        if key in obs:
            grip = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            candidates.append(grip[:1])
            break
    out = np.concatenate(candidates).astype(np.float32)
    if out.shape[0] < 8:
        out = np.pad(out, (0, 8 - out.shape[0]))
    return out[:8]


def _frame_from_obs(obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    frame = obs.get(camera_name)
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame[::-1]


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return str(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    _prepare_libero_paths(args.libero_root, output_dir)

    try:
        benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_api()
    except Exception as exc:
        result = {
            "status": "skipped",
            "reason": f"LIBERO import failed: {type(exc).__name__}: {exc}",
            "checkpoint": str(args.checkpoint),
        }
        _write_json(output_dir / "result.json", result)
        return result

    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model, train_args = _load_model(args.checkpoint, device)
        obs_full_state = bool(train_args.get("obs_full_state"))
        horizon = int(train_args.get("action_horizon", 8))

        total_successes = 0
        total_trials = 0
        videos: list[str] = []
        task_results: list[dict[str, Any]] = []
        started = time.time()

        for benchmark_name in _benchmark_names(args.benchmark):
            task_suite = benchmark_dict[benchmark_name]()
            for task_id in _task_ids(task_suite, args.task_id):
                task = task_suite.get_task(task_id)
                bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                init_states = task_suite.get_task_init_states(task_id)
                env = OffScreenRenderEnv(
                    bddl_file_name=str(bddl_file),
                    camera_heights=args.camera_size,
                    camera_widths=args.camera_size,
                )
                env.seed(args.seed)

                successes = 0
                trial_stats: list[dict[str, Any]] = []
                with torch.no_grad():
                    for trial in range(args.trials):
                        env.reset()
                        obs = env.set_init_state(init_states[trial % len(init_states)])
                        frames: list[np.ndarray] = []
                        done = False
                        steps = 0
                        for _ in range(args.settle_steps):
                            obs, _, done, _ = env.step(np.zeros(7, dtype=np.float32))
                            frame = _frame_from_obs(obs, args.camera_name)
                            if frame is not None:
                                frames.append(frame)

                        while steps < args.max_steps and not done:
                            obs_vec = _obs_vector(obs, obs_full_state)
                            obs_t = torch.as_tensor(obs_vec, device=device, dtype=torch.float32).unsqueeze(0)
                            chunk = model(obs_t).squeeze(0).detach().cpu().numpy()
                            for action in chunk[:horizon]:
                                obs, _, done, info = env.step(action.astype(np.float32))
                                steps += 1
                                frame = _frame_from_obs(obs, args.camera_name)
                                if frame is not None:
                                    frames.append(frame)
                                done = bool(done or info.get("success", False))
                                if done or steps >= args.max_steps:
                                    break

                        successes += int(done)
                        video_path = (
                            output_dir
                            / "videos"
                            / benchmark_name
                            / f"task{task_id:03d}_trial{trial:03d}.mp4"
                        )
                        saved_video = _write_video(video_path, frames, args.fps) if args.save_video else None
                        if saved_video is not None:
                            videos.append(saved_video)
                        trial_stats.append(
                            {
                                "trial": trial,
                                "success": bool(done),
                                "steps": steps,
                                "video": saved_video,
                            }
                        )
                env.close()
                total_successes += successes
                total_trials += args.trials
                task_results.append(
                    {
                        "benchmark": benchmark_name,
                        "task_id": task_id,
                        "task_name": task.name,
                        "language": task.language,
                        "trials": args.trials,
                        "successes": successes,
                        "success_rate": successes / max(args.trials, 1),
                        "trial_stats": trial_stats,
                    }
                )

        result = {
            "status": "completed",
            "checkpoint": str(args.checkpoint),
            "benchmark": args.benchmark,
            "task_id": args.task_id,
            "trials_per_task": args.trials,
            "total_trials": total_trials,
            "successes": total_successes,
            "success_rate": total_successes / max(total_trials, 1),
            "videos": videos,
            "task_results": task_results,
            "elapsed_sec": time.time() - started,
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "checkpoint": str(args.checkpoint),
            "benchmark": args.benchmark,
            "task_id": args.task_id,
        }

    _write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a New-IL checkpoint in LIBERO.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="libero_object")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--camera-name", default="agentview_image")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--libero-root", type=Path, default=os.environ.get("LIBERO_ROOT"))
    args = parser.parse_args()

    result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
