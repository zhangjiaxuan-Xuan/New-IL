from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from new_il.integrations.openpi import (
    LIBERO_DUMMY_ACTION,
    OpenPILiberoConfig,
    OpenPIWebsocketPolicy,
    build_openpi_libero_payload,
    libero_state_for_openpi,
)


_LIBERO_INIT_STATES_CACHE: dict[tuple[str, int], Any] = {}


class RolloutPolicy(Protocol):
    def reset(self) -> None: ...

    def select_action(self, raw_obs: dict[str, Any], formatted_obs: dict[str, Any], language: str) -> np.ndarray: ...


@dataclass(frozen=True)
class LiberoRolloutConfig:
    task_suite_name: str = "libero_spatial"
    max_steps: int = 220
    settle_steps: int = 10
    resolution: int = 256
    fps: int = 10
    save_video: bool = True
    save_traj: bool = True


@dataclass
class EpisodeResult:
    task_name: str
    success: bool
    n_steps: int
    traj_npz_path: Path | None
    video_path: Path | None
    action_health: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0


def _import_libero_api():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def prepare_libero_paths(libero_root: Path | None, output_dir: Path) -> None:
    if libero_root is None:
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    libero_root = libero_root.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[3]
    path_candidates = [
        project_root / "third_party" / "robosuite",
        project_root / "third_party" / "robosuite-mem",
        project_root / "third_party",
        libero_root,
    ]
    for candidate in reversed(path_candidates):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

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


def _libero_init_states(task_suite: str, task_idx: int) -> Any:
    key = (task_suite, int(task_idx))
    if key not in _LIBERO_INIT_STATES_CACHE:
        import torch
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark_dict

        suite = get_benchmark_dict()[task_suite]()
        task = suite.get_task(int(task_idx))
        init_states_path = (
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        _LIBERO_INIT_STATES_CACHE[key] = torch.load(init_states_path, weights_only=False)
    return _LIBERO_INIT_STATES_CACHE[key]


def _format_lerobot_observation(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert raw LIBERO observations to the LeRobot LIBERO policy schema.

    This mirrors Mem's rollout format so collected NPZ files remain compatible
    with downstream B_sup / PATCS tooling.
    """

    state = libero_state_for_openpi(raw_obs)
    return {
        "pixels": {
            "image": np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1], dtype=np.uint8),
            "image2": np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8),
        },
        "agent_pos": state.astype(np.float32, copy=False),
    }


def _text_features(language: str, length: int) -> np.ndarray:
    vec = np.frombuffer(language.encode("utf-8"), dtype=np.uint8).astype(np.float32)
    if vec.size == 0:
        base = np.zeros((8,), dtype=np.float32)
    else:
        chunks = np.array_split(vec, 8)
        base = np.array([chunk.mean() if len(chunk) else 0.0 for chunk in chunks], dtype=np.float32) / 255.0
    return np.repeat(base[None, :], length, axis=0)


def _vision_features(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        return np.zeros((0, 8), dtype=np.float32)
    feats = []
    for image in images:
        rgb = np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0)
        rgb = rgb / (np.linalg.norm(rgb) + 1e-6)
        feats.append(np.pad(rgb, (0, 5))[:8])
    return np.asarray(feats, dtype=np.float32)


def _save_trajectory_npz(traj_buffer: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.stack(traj_buffer["actions"]).astype(np.float32)
    images = np.stack(traj_buffer["images"]).astype(np.uint8) if traj_buffer.get("images") else None
    arrays: dict[str, np.ndarray] = {
        "language": np.array(traj_buffer["language"]),
        "text": _text_features(str(traj_buffer["language"]), actions.shape[0]),
        "vision": _vision_features(traj_buffer.get("images", [])),
        "actions": actions,
        "success": np.array(bool(traj_buffer["success"])),
        "metadata_json": np.asarray(json.dumps(traj_buffer.get("metadata", {}), sort_keys=True)),
    }
    if images is not None:
        arrays["observation.images.image"] = images
    if traj_buffer.get("images2"):
        arrays["observation.images.image2"] = np.stack(traj_buffer["images2"]).astype(np.uint8)
    if traj_buffer.get("state"):
        arrays["observation.state"] = np.stack(traj_buffer["state"]).astype(np.float32)
    np.savez_compressed(str(output_path), **arrays)


def _task_video_dirname(language: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", language.strip().lower()).strip("_")
    return name or "unknown_task"


def _save_episode_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, [np.ascontiguousarray(frame) for frame in frames], fps=fps)


def action_health_summary(actions: list[np.ndarray], exceptions: int = 0) -> dict[str, Any]:
    if not actions:
        return {
            "n_actions": 0,
            "select_action_exceptions": int(exceptions),
            "all_zero": True,
            "stable": False,
        }
    arr = np.asarray(actions, dtype=np.float32)
    mean_abs = float(np.abs(arr).mean())
    std = float(arr.std())
    repeated = np.all(np.isclose(np.diff(arr, axis=0), 0.0, atol=1e-6), axis=1) if len(arr) > 1 else []
    return {
        "n_actions": int(arr.shape[0]),
        "select_action_exceptions": int(exceptions),
        "mean_abs": mean_abs,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "xyz_norm_mean": float(np.linalg.norm(arr[:, :3], axis=1).mean()),
        "all_zero": bool(np.allclose(arr, 0.0, atol=1e-8)),
        "repeated_action_rate": float(np.asarray(repeated, dtype=np.float32).mean()) if len(arr) > 1 else 0.0,
        "stable": exceptions == 0 and mean_abs > 1e-5 and std > 1e-5,
    }


class OpenPIChunkPolicy:
    """OpenPI websocket policy backend with local action-chunk caching."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        replan_steps: int = 5,
        resize_size: int = 224,
        api_key: str | None = None,
    ) -> None:
        self.client = OpenPIWebsocketPolicy(host=host, port=port, api_key=api_key)
        self.replan_steps = int(replan_steps)
        self.config = OpenPILiberoConfig(resize_size=resize_size)
        self._action_plan: list[np.ndarray] = []

    def reset(self) -> None:
        self._action_plan.clear()

    def select_action(self, raw_obs: dict[str, Any], formatted_obs: dict[str, Any], language: str) -> np.ndarray:
        del formatted_obs
        if not self._action_plan:
            payload = build_openpi_libero_payload(raw_obs, language, self.config)
            chunk = self.client.action_chunk(payload)
            if len(chunk) < self.replan_steps:
                raise ValueError(f"OpenPI returned {len(chunk)} actions, fewer than replan_steps={self.replan_steps}")
            self._action_plan.extend(np.asarray(action, dtype=np.float32) for action in chunk[: self.replan_steps])
        return self._action_plan.pop(0)


def run_one_episode(
    *,
    config: LiberoRolloutConfig,
    task: Any,
    language: str,
    episode_idx: int,
    seed: int,
    policy: RolloutPolicy,
    output_dir: Path,
    task_idx: int = 0,
    round_idx: int = 0,
) -> tuple[EpisodeResult, Path | None, np.ndarray | None]:
    """Run one LIBERO episode with a pluggable policy backend.

    The environment stepping, observation formatting, trajectory NPZ schema, and
    video layout follow Mem's rollout design. Only action generation is
    delegated to the supplied policy backend.
    """

    _, get_libero_path, OffScreenRenderEnv = _import_libero_api()
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=config.resolution,
        camera_widths=config.resolution,
    )
    init_states = _libero_init_states(config.task_suite_name, task_idx)
    started = time.time()
    select_action_exceptions = 0
    success = False
    video_path: Path | None = None
    traj_path: Path | None = None
    buf: dict[str, Any] = {
        "language": language,
        "actions": [],
        "images": [],
        "images2": [],
        "state": [],
        "success": False,
        "metadata": {
            "schema_version": 1,
            "source": "new_il_mem_style_rollout",
            "task_suite_name": config.task_suite_name,
            "task_idx": int(task_idx),
            "episode_idx": int(episode_idx),
            "seed": int(seed),
            "round_idx": int(round_idx),
        },
    }
    video_frames: list[np.ndarray] = []

    try:
        policy.reset()
        env.seed(seed)
        env.set_init_state(init_states[episode_idx % len(init_states)])
        obs = env.reset()
        for _ in range(config.settle_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        for _step_t in range(config.max_steps):
            formatted_obs = _format_lerobot_observation(obs)
            buf["images"].append(formatted_obs["pixels"]["image"])
            buf["images2"].append(formatted_obs["pixels"]["image2"])
            buf["state"].append(formatted_obs["agent_pos"])

            if config.save_video:
                frame = np.concatenate(
                    [formatted_obs["pixels"]["image"], formatted_obs["pixels"]["image2"]],
                    axis=1,
                ).astype(np.uint8)
                video_frames.append(np.ascontiguousarray(frame))

            try:
                action = policy.select_action(obs, formatted_obs, language)
            except Exception:
                select_action_exceptions += 1
                raise
            action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
            if action.shape[0] != 7:
                raise ValueError(f"expected action shape [7], got {action.shape}")
            buf["actions"].append(action)
            obs, done, _, info = _step_env(env, action)
            if bool(getattr(env, "check_success", lambda: False)()) or bool(info.get("success", False)):
                success = True
                break
            if done:
                break
    finally:
        env.close()

    buf["success"] = success
    actions_arr = np.asarray(buf["actions"], dtype=np.float32) if buf["actions"] else None
    ep_name = f"round{round_idx:02d}_task{task_idx:02d}_ep{episode_idx:04d}_seed{seed}"
    if config.save_traj and actions_arr is not None:
        root = "success_rollouts" if success else "failed_rollouts"
        traj_path = output_dir / root / f"{ep_name}.npz"
        _save_trajectory_npz(buf, traj_path)
    if config.save_video and video_frames:
        tag = "success" if success else "fail"
        video_path = output_dir / "videos" / _task_video_dirname(language) / f"{ep_name}_{tag}.mp4"
        _save_episode_video(video_frames, video_path, config.fps)

    result = EpisodeResult(
        task_name=language,
        success=success,
        n_steps=int(len(buf["actions"])),
        traj_npz_path=traj_path if success else None,
        video_path=video_path,
        action_health=action_health_summary(buf["actions"], select_action_exceptions),
        elapsed_sec=time.time() - started,
    )
    return result, (traj_path if success else None), actions_arr


def _step_env(env: Any, action: np.ndarray) -> tuple[dict[str, Any], bool, float, dict[str, Any]]:
    obs, reward, done, info = env.step(action)
    return obs, bool(done), float(reward), dict(info)


def _max_steps(task_suite_name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(task_suite_name, 400)


def openpi_rollout_main() -> None:
    parser = argparse.ArgumentParser(description="Run one Mem-style LIBERO rollout against an OpenPI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--libero-root", type=Path, default=Path("third_party/LIBERO"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/openpi_libero_smoke"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    prepare_libero_paths(args.libero_root, args.output_dir)
    benchmark, _, _ = _import_libero_api()
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    policy = OpenPIChunkPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        replan_steps=args.replan_steps,
        resize_size=args.resize_size,
    )
    result, traj_path, _actions = run_one_episode(
        config=LiberoRolloutConfig(
            task_suite_name=args.task_suite_name,
            max_steps=args.max_steps or _max_steps(args.task_suite_name),
            settle_steps=args.settle_steps,
            resolution=args.camera_size,
            fps=args.fps,
            save_video=not args.no_video,
            save_traj=True,
        ),
        task=task,
        language=str(task.language),
        episode_idx=args.episode_idx,
        seed=args.seed,
        policy=policy,
        output_dir=args.output_dir,
        task_idx=args.task_id,
    )
    payload = {
        "status": "ok",
        "success": result.success,
        "steps": result.n_steps,
        "traj_npz_path": str(traj_path) if traj_path else None,
        "video_path": str(result.video_path) if result.video_path else None,
        "action_health": result.action_health,
        "elapsed_sec": result.elapsed_sec,
    }
    (args.output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
