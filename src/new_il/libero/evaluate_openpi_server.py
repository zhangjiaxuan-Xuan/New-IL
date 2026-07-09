from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from new_il.integrations.openpi import (
    LIBERO_DUMMY_ACTION,
    OpenPILiberoConfig,
    OpenPIWebsocketPolicy,
    build_openpi_libero_payload,
    libero_state_for_openpi,
    openpi_libero_image,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _import_libero_api():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def _max_steps(task_suite_name: str) -> int:
    defaults = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in defaults:
        raise ValueError(f"unknown LIBERO task suite: {task_suite_name}")
    return defaults[task_suite_name]


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [np.ascontiguousarray(frame) for frame in frames], fps=fps)
    return str(path)


def _frame_from_obs(obs: dict[str, Any], key: str) -> np.ndarray:
    return openpi_libero_image(obs[key])


def _trajectory_path(output_dir: Path, task_suite_name: str, task_id: int, trial: int, success: bool) -> Path:
    suffix = "success" if success else "failure"
    return output_dir / "trajectories" / f"{task_suite_name}_task{task_id:03d}_trial{trial:03d}_{suffix}.npz"


def _save_trajectory(
    path: Path,
    *,
    task_suite_name: str,
    task_id: int,
    task_name: str,
    language: str,
    trial: int,
    seed: int,
    success: bool,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    agent_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    policy_images: list[np.ndarray],
    policy_wrist_images: list[np.ndarray],
) -> str | None:
    if not actions:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "source": "new_il_openpi_libero_rollout",
        "task_suite_name": task_suite_name,
        "task_id": int(task_id),
        "task_name": task_name,
        "language": language,
        "trial": int(trial),
        "seed": int(seed),
        "success": bool(success),
        "num_steps": int(len(actions)),
    }
    np.savez_compressed(
        path,
        schema_version=np.array(1, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        task_suite_name=np.asarray(task_suite_name),
        task_id=np.array(task_id, dtype=np.int32),
        task_name=np.asarray(task_name),
        language=np.asarray(language),
        trial=np.array(trial, dtype=np.int32),
        seed=np.array(seed, dtype=np.int32),
        success=np.array(success, dtype=bool),
        observation_state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        observation_images_image=np.asarray(agent_images, dtype=np.uint8),
        observation_images_image2=np.asarray(wrist_images, dtype=np.uint8),
        policy_observation_image=np.asarray(policy_images, dtype=np.uint8),
        policy_observation_wrist_image=np.asarray(policy_wrist_images, dtype=np.uint8),
    )
    return str(path)


def _action_health(actions: list[np.ndarray]) -> dict[str, Any]:
    if not actions:
        return {"count": 0}
    arr = np.asarray(actions, dtype=np.float32)
    delta_xyz = arr[:, :3]
    repeated = np.all(np.isclose(np.diff(arr, axis=0), 0.0, atol=1e-6), axis=1) if len(arr) > 1 else []
    return {
        "count": int(arr.shape[0]),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "xyz_norm_mean": float(np.linalg.norm(delta_xyz, axis=1).mean()),
        "xyz_norm_max": float(np.linalg.norm(delta_xyz, axis=1).max()),
        "gripper_min": float(arr[:, -1].min()),
        "gripper_max": float(arr[:, -1].max()),
        "all_zero_rate": float(np.all(np.isclose(arr, 0.0, atol=1e-6), axis=1).mean()),
        "repeated_action_rate": float(np.asarray(repeated, dtype=np.float32).mean()) if len(arr) > 1 else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    _prepare_libero_paths(args.libero_root, output_dir)
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_api()

    policy = OpenPIWebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    env.seed(args.seed)

    config = OpenPILiberoConfig(resize_size=args.resize_size)
    max_steps = args.max_steps or _max_steps(args.task_suite_name)
    trials: list[dict[str, Any]] = []
    started = time.time()
    try:
        for trial in range(args.trials):
            env.reset()
            obs = env.set_init_state(init_states[trial % len(init_states)])
            done = False
            steps = 0
            action_plan: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            frames: list[np.ndarray] = []
            states: list[np.ndarray] = []
            agent_images: list[np.ndarray] = []
            wrist_images: list[np.ndarray] = []
            policy_images: list[np.ndarray] = []
            policy_wrist_images: list[np.ndarray] = []

            for _ in range(args.settle_steps):
                obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
                done = bool(done or info.get("success", False))

            while steps < max_steps and not done:
                if not action_plan:
                    payload = build_openpi_libero_payload(obs, task.language, config)
                    frames.append(payload["observation/image"])
                    policy_images.append(payload["observation/image"])
                    policy_wrist_images.append(payload["observation/wrist_image"])
                    chunk = policy.action_chunk(payload)
                    if len(chunk) < args.replan_steps:
                        raise ValueError(
                            f"OpenPI returned {len(chunk)} actions, fewer than replan_steps={args.replan_steps}"
                        )
                    action_plan.extend(np.asarray(action, dtype=np.float32) for action in chunk[: args.replan_steps])

                action = action_plan.pop(0)
                states.append(libero_state_for_openpi(obs))
                agent_images.append(_frame_from_obs(obs, config.agent_image_key))
                wrist_images.append(_frame_from_obs(obs, config.wrist_image_key))
                obs, _, done, info = env.step(action)
                done = bool(done or info.get("success", False))
                actions.append(action)
                steps += 1

            video = None
            if args.save_video:
                suffix = "success" if done else "failure"
                video = _write_video(
                    output_dir / "videos" / f"{args.task_suite_name}_task{args.task_id:03d}_{trial:03d}_{suffix}.mp4",
                    frames,
                    args.fps,
                )
            trajectory = _save_trajectory(
                _trajectory_path(output_dir, args.task_suite_name, args.task_id, trial, bool(done)),
                task_suite_name=args.task_suite_name,
                task_id=args.task_id,
                task_name=task.name,
                language=task.language,
                trial=trial,
                seed=args.seed,
                success=bool(done),
                states=states,
                actions=actions,
                agent_images=agent_images,
                wrist_images=wrist_images,
                policy_images=policy_images,
                policy_wrist_images=policy_wrist_images,
            )
            trials.append(
                {
                    "trial": trial,
                    "success": bool(done),
                    "steps": steps,
                    "video": video,
                    "trajectory": trajectory,
                    "action_health": _action_health(actions),
                }
            )
    finally:
        env.close()

    result = {
        "status": "ok",
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "task_name": task.name,
        "language": task.language,
        "policy_uri": policy.uri,
        "policy_metadata": policy.metadata,
        "trials": trials,
        "success_rate": sum(int(t["success"]) for t in trials) / max(len(trials), 1),
        "elapsed_sec": time.time() - started,
    }
    _write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a running OpenPI policy server on LIBERO.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--libero-root", type=Path, default=Path("third_party/LIBERO"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/openpi_libero_eval"))
    parser.add_argument("--save-video", action="store_true", default=True)
    result = evaluate(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
