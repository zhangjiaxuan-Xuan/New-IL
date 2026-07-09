"""LIBERO action-chunk dataset backed by per-task HDF5 files and PATCS artifacts.

Each item contains:
  obs          float32 [obs_dim]         current ee_pos (and optionally full ee_states)
  action_chunk float32 [horizon, 7]      GT action chunk starting at this step
  ee_pos_seq   float32 [horizon, 3]      GT ee_pos for the same window (PATCS reference)
  stage        int                       which gripper-defined stage this step is in
  rho_start    float32                   normalized progress within the stage

Only demos present in the artifact are used so that stage_boundaries and rho are
always well-defined. HDF5 demos beyond the artifact's num_demos are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError as exc:
    raise SystemExit("torch is required. Run: uv sync --extra train") from exc

from new_il.training.patcs_loss import PatcsArtifact, load_patcs_artifact


def _action_health_from_sequences(actions_list: list[np.ndarray]) -> dict[str, float | int]:
    if not actions_list:
        return {
            "num_sequences": 0,
            "num_actions": 0,
            "action_mean_abs": 0.0,
            "action_std": 0.0,
            "xyz_norm_mean": 0.0,
            "all_zero_rate": 1.0,
            "repeated_action_rate": 1.0,
        }
    actions = np.concatenate([np.asarray(item, dtype=np.float32) for item in actions_list], axis=0)
    all_zero = [float(np.allclose(item, 0.0, atol=1e-8)) for item in actions_list]
    repeated = []
    for item in actions_list:
        if len(item) <= 1:
            repeated.append(0.0)
        else:
            repeated.append(float(np.all(np.isclose(np.diff(item, axis=0), 0.0, atol=1e-6), axis=1).mean()))
    return {
        "num_sequences": len(actions_list),
        "num_actions": int(actions.shape[0]),
        "action_mean_abs": float(np.abs(actions).mean()),
        "action_std": float(actions.std()),
        "xyz_norm_mean": float(np.linalg.norm(actions[:, :3], axis=1).mean()),
        "all_zero_rate": float(np.mean(all_zero)),
        "repeated_action_rate": float(np.mean(repeated)),
    }


@dataclass(frozen=True)
class ChunkItem:
    obs: np.ndarray           # [obs_dim]
    action_chunk: np.ndarray  # [H, 7]
    ee_pos_seq: np.ndarray    # [H, 3]  GT positions for PATCS
    prev_action: np.ndarray   # [7] previous executed action for boundary smoothness
    stage: int
    rho_start: float
    task_name: str


def _find_stage_and_rho(
    step: int,
    stage_boundaries: np.ndarray,  # [num_stages, 2]
) -> tuple[int, float]:
    """Return (stage_idx, rho_start) for a given absolute step index."""
    num_stages = stage_boundaries.shape[0]
    for s in range(num_stages):
        start, end = int(stage_boundaries[s, 0]), int(stage_boundaries[s, 1])
        if start <= step < end:
            width = max(end - start, 1)
            return s, float(step - start) / width
    # Step is at or past the last boundary end; clamp to last stage at rho=1
    last = num_stages - 1
    return last, 1.0



@dataclass
class _TaskData:
    """Per-task in-memory cache: only the artifact-matched demos."""

    task_name: str
    artifact: PatcsArtifact
    # List of (ee_pos [T,3], ee_states [T,8], actions [T,7], stage_boundaries [S,2])
    demos: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

    @staticmethod
    def load(hdf5_path: Path, artifact_path: Path, obs_full_state: bool = False) -> "_TaskData":
        artifact = load_patcs_artifact(artifact_path)

        # demo_ids are bytes stored in the npz; reload to get them
        raw = np.load(artifact_path, allow_pickle=False)
        demo_ids: list[str] = [b.decode() for b in raw["demo_ids"]]
        # stage_boundaries: [num_demos, num_stages, 2]
        stage_boundaries_all: np.ndarray = raw["stage_boundaries"].astype(np.int32)

        demos = []
        with h5py.File(hdf5_path, "r") as hf:
            for demo_idx, demo_name in enumerate(demo_ids):
                if demo_name not in hf["data"]:
                    continue
                grp = hf["data"][demo_name]
                ee_pos = np.asarray(grp["obs"]["ee_pos"], dtype=np.float32)     # [T, 3]
                ee_states = np.asarray(grp["obs"]["ee_states"], dtype=np.float32)  # [T, 8]
                actions = np.asarray(grp["actions"], dtype=np.float32)          # [T, 7]
                boundaries = stage_boundaries_all[demo_idx]                      # [S, 2]
                demos.append((ee_pos, ee_states, actions, boundaries))

        task_name = hdf5_path.stem
        return _TaskData(task_name=task_name, artifact=artifact, demos=demos)

    def iter_chunks(self, horizon: int, obs_full_state: bool) -> Iterator[ChunkItem]:
        for ee_pos, ee_states, actions, boundaries in self.demos:
            T = len(actions)
            obs_array = ee_states if obs_full_state else ee_pos  # [T, D_obs]
            for t in range(T - horizon):
                stage, rho = _find_stage_and_rho(t, boundaries)
                prev_action = actions[t - 1] if t > 0 else np.zeros((actions.shape[1],), dtype=np.float32)
                yield ChunkItem(
                    obs=obs_array[t].copy(),
                    action_chunk=actions[t : t + horizon].copy(),
                    ee_pos_seq=ee_pos[t : t + horizon].copy(),
                    prev_action=prev_action.copy(),
                    stage=stage,
                    rho_start=rho,
                    task_name=self.task_name,
                )


class LiberoChunkDataset(Dataset):
    """PyTorch Dataset of action chunks from LIBERO RLDS-converted HDF5 files.

    Args:
        hdf5_dir: directory containing per-task .hdf5 files for one suite.
        artifact_dir: directory containing matching *_patcs.npz files.
        horizon: number of steps per action chunk (default 8).
        obs_full_state: if True use ee_states [8] as obs instead of ee_pos [3].
        max_tasks: cap number of tasks (useful for smoke testing).
    """

    def __init__(
        self,
        hdf5_dir: Path | str,
        artifact_dir: Path | str,
        horizon: int = 8,
        obs_full_state: bool = False,
        max_tasks: int | None = None,
        verbose: bool = True,
    ) -> None:
        hdf5_dir = Path(hdf5_dir)
        artifact_dir = Path(artifact_dir)

        self._horizon = horizon
        self._obs_full_state = obs_full_state
        self._items: list[ChunkItem] = []
        self._artifacts: dict[str, PatcsArtifact] = {}

        hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
        if not hdf5_files:
            hdf5_files = sorted(hdf5_dir.glob("*/*.hdf5"))
        if max_tasks is not None:
            hdf5_files = hdf5_files[:max_tasks]
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files found in {hdf5_dir}")

        loaded_tasks = 0
        for hdf5_path in hdf5_files:
            artifact_path = _find_artifact(hdf5_path, artifact_dir)
            if artifact_path is None:
                continue
            task_data = _TaskData.load(hdf5_path, artifact_path, obs_full_state)
            self._artifacts[task_data.task_name] = task_data.artifact
            self._items.extend(task_data.iter_chunks(horizon, obs_full_state))
            loaded_tasks += 1

        if not self._items:
            raise RuntimeError(
                f"No training items found. Check that HDF5 and artifact directories match.\n"
                f"HDF5: {hdf5_dir}\nArtifacts: {artifact_dir}"
            )

        self._obs_dim = self._items[0].obs.shape[0]
        if verbose:
            print(
                f"LiberoChunkDataset: {loaded_tasks} tasks, "
                f"{len(self._items):,} chunks, obs_dim={self._obs_dim}, horizon={horizon}, "
                f"hdf5_root={hdf5_dir}, artifact_root={artifact_dir}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        item = self._items[idx]
        return {
            "obs": item.obs,
            "action_chunk": item.action_chunk,
            "ee_pos_seq": item.ee_pos_seq,
            "prev_action": item.prev_action,
            "stage": np.int64(item.stage),
            "rho_start": np.float32(item.rho_start),
            "task_name": item.task_name,
        }

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return 7

    def get_artifact(self, task_name: str) -> PatcsArtifact:
        return self._artifacts[task_name]


@dataclass
class _RolloutTaskData:
    """Per-task rollout cache built from cleaned by_task directories."""

    task_name: str
    artifact: PatcsArtifact
    demos: list[tuple[np.ndarray, np.ndarray, np.ndarray]]

    @staticmethod
    def load(task_dir: Path, artifact_path: Path, obs_full_state: bool = False) -> "_RolloutTaskData":
        del obs_full_state
        artifact = load_patcs_artifact(artifact_path)
        raw = np.load(artifact_path, allow_pickle=False)
        demo_ids: list[str] = [b.decode() for b in raw["demo_ids"]]
        stage_boundaries_all: np.ndarray = raw["stage_boundaries"].astype(np.int32)

        demos = []
        success_dir = task_dir / "success_rollouts"
        for demo_idx, demo_id in enumerate(demo_ids):
            npz_path = success_dir / f"{demo_id}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=False)
            state = np.asarray(data["observation.state"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            if state.shape[0] != actions.shape[0]:
                continue
            ee_pos = state[:, :3].astype(np.float32, copy=False)
            boundaries = stage_boundaries_all[demo_idx]
            demos.append((ee_pos, state, actions, boundaries))

        suite = task_dir.parent.name
        task_name = f"{suite}_{task_dir.name}"
        return _RolloutTaskData(task_name=task_name, artifact=artifact, demos=demos)

    def iter_chunks(self, horizon: int, obs_full_state: bool) -> Iterator[ChunkItem]:
        for ee_pos, state, actions, boundaries in self.demos:
            T = len(actions)
            obs_array = state if obs_full_state else ee_pos
            for t in range(T - horizon):
                stage, rho = _find_stage_and_rho(t, boundaries)
                prev_action = actions[t - 1] if t > 0 else np.zeros((actions.shape[1],), dtype=np.float32)
                yield ChunkItem(
                    obs=obs_array[t].copy(),
                    action_chunk=actions[t : t + horizon].copy(),
                    ee_pos_seq=ee_pos[t : t + horizon].copy(),
                    prev_action=prev_action.copy(),
                    stage=stage,
                    rho_start=rho,
                    task_name=self.task_name,
                )


class RolloutChunkDataset(Dataset):
    """Action-chunk dataset backed by cleaned rollout NPZ directories.

    Expected layout:
      rollout_root/<suite>/task_XX/success_rollouts/*.npz

    This is the direct training-side consumer for ``new-il-rollout-manifest`` output.
    It uses matching PATCS artifacts to recover stage boundaries and rho_start.
    """

    def __init__(
        self,
        rollout_root: Path | str,
        artifact_dir: Path | str,
        horizon: int = 8,
        obs_full_state: bool = False,
        max_tasks: int | None = None,
        verbose: bool = True,
    ) -> None:
        rollout_root = Path(rollout_root)
        artifact_dir = Path(artifact_dir)
        self._horizon = horizon
        self._obs_full_state = obs_full_state
        self._items: list[ChunkItem] = []
        self._artifacts: dict[str, PatcsArtifact] = {}

        if (rollout_root / "success_rollouts").exists():
            task_dirs = [rollout_root]
        else:
            task_dirs = sorted(path for path in rollout_root.glob("*/*") if (path / "success_rollouts").exists())
        if max_tasks is not None:
            task_dirs = task_dirs[:max_tasks]
        if not task_dirs:
            raise FileNotFoundError(f"No cleaned rollout task dirs found in {rollout_root}")

        loaded_tasks = 0
        for task_dir in task_dirs:
            task_name = f"{task_dir.parent.name}_{task_dir.name}"
            artifact_path = _find_artifact_for_task_name(task_name, task_dir, artifact_dir)
            if artifact_path is None:
                continue
            task_data = _RolloutTaskData.load(task_dir, artifact_path, obs_full_state)
            self._artifacts[task_data.task_name] = task_data.artifact
            self._items.extend(task_data.iter_chunks(horizon, obs_full_state))
            loaded_tasks += 1

        if not self._items:
            raise RuntimeError(
                f"No rollout training items found. Check rollout/artifact match.\n"
                f"Rollouts: {rollout_root}\nArtifacts: {artifact_dir}"
            )

        self._obs_dim = self._items[0].obs.shape[0]
        if verbose:
            print(
                f"RolloutChunkDataset: {loaded_tasks} tasks, "
                f"{len(self._items):,} chunks, obs_dim={self._obs_dim}, horizon={horizon}, "
                f"rollout_root={rollout_root}, artifact_root={artifact_dir}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        item = self._items[idx]
        return {
            "obs": item.obs,
            "action_chunk": item.action_chunk,
            "ee_pos_seq": item.ee_pos_seq,
            "prev_action": item.prev_action,
            "stage": np.int64(item.stage),
            "rho_start": np.float32(item.rho_start),
            "task_name": item.task_name,
        }

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return 7

    def get_artifact(self, task_name: str) -> PatcsArtifact:
        return self._artifacts[task_name]

    def health_summary(self) -> dict[str, float | int]:
        return _action_health_from_sequences([item.action_chunk for item in self._items])


@dataclass
class TrajectoryWorkerState:
    """Mutable state for simplified dynamic trajectory supervision."""

    demo_idx: int
    obs_index: int
    previous_chunk: np.ndarray


@dataclass
class RolloutTrajectory:
    task_name: str
    artifact: PatcsArtifact
    ee_pos: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    stage_boundaries: np.ndarray
    demo_id: str


class RolloutTrajectoryDataset:
    """Full rollout trajectories for dynamic PATCS worker-style training.

    This is intentionally not a map-style Dataset. The training loop keeps a
    small set of worker states and asks for chunks at each worker's current
    ``obs_index``. After the policy predicts a chunk, the trainer advances the
    worker's ``obs_index`` by matching the predicted terminal ee position back
    to the same demonstration trajectory.
    """

    def __init__(
        self,
        rollout_root: Path | str,
        artifact_dir: Path | str,
        horizon: int = 8,
        obs_full_state: bool = False,
        max_tasks: int | None = None,
        verbose: bool = True,
    ) -> None:
        rollout_root = Path(rollout_root)
        artifact_dir = Path(artifact_dir)
        self.horizon = int(horizon)
        self.obs_full_state = bool(obs_full_state)
        self.trajectories: list[RolloutTrajectory] = []
        self._artifacts: dict[str, PatcsArtifact] = {}

        if (rollout_root / "success_rollouts").exists():
            task_dirs = [rollout_root]
        else:
            task_dirs = sorted(path for path in rollout_root.glob("*/*") if (path / "success_rollouts").exists())
        if max_tasks is not None:
            task_dirs = task_dirs[:max_tasks]
        if not task_dirs:
            raise FileNotFoundError(f"No cleaned rollout task dirs found in {rollout_root}")

        for task_dir in task_dirs:
            task_name = f"{task_dir.parent.name}_{task_dir.name}"
            artifact_path = _find_artifact_for_task_name(task_name, task_dir, artifact_dir)
            if artifact_path is None:
                continue
            artifact = load_patcs_artifact(artifact_path)
            raw = np.load(artifact_path, allow_pickle=False)
            demo_ids: list[str] = [b.decode() for b in raw["demo_ids"]]
            boundaries_all = raw["stage_boundaries"].astype(np.int32)
            self._artifacts[task_name] = artifact
            success_dir = task_dir / "success_rollouts"
            for demo_idx, demo_id in enumerate(demo_ids):
                npz_path = success_dir / f"{demo_id}.npz"
                if not npz_path.exists():
                    continue
                data = np.load(npz_path, allow_pickle=False)
                state = np.asarray(data["observation.state"], dtype=np.float32)
                actions = np.asarray(data["actions"], dtype=np.float32)
                if len(actions) <= self.horizon or state.shape[0] != actions.shape[0]:
                    continue
                ee_pos = state[:, :3].astype(np.float32, copy=False)
                self.trajectories.append(
                    RolloutTrajectory(
                        task_name=task_name,
                        artifact=artifact,
                        ee_pos=ee_pos,
                        state=state,
                        actions=actions,
                        stage_boundaries=boundaries_all[demo_idx],
                        demo_id=demo_id,
                    )
                )

        if not self.trajectories:
            raise RuntimeError(
                f"No rollout trajectories found. Check rollout/artifact match.\n"
                f"Rollouts: {rollout_root}\nArtifacts: {artifact_dir}"
            )
        self._obs_dim = 8 if obs_full_state else 3
        if verbose:
            print(
                f"RolloutTrajectoryDataset: {len(self.trajectories)} trajectories, "
                f"obs_dim={self._obs_dim}, horizon={self.horizon}, "
                f"rollout_root={rollout_root}, artifact_root={artifact_dir}",
                flush=True,
            )

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def artifacts(self) -> dict[str, PatcsArtifact]:
        return self._artifacts

    def health_summary(self) -> dict[str, float | int]:
        summary = _action_health_from_sequences([traj.actions for traj in self.trajectories])
        stage_oob = 0
        rho_oob = 0
        checked = 0
        for traj in self.trajectories:
            for step in (0, max(0, len(traj.actions) // 2), max(0, len(traj.actions) - self.horizon - 1)):
                stage, rho = _find_stage_and_rho(step, traj.stage_boundaries)
                stage_oob += int(stage < 0 or stage >= traj.stage_boundaries.shape[0])
                rho_oob += int(rho < 0.0 or rho > 1.0)
                checked += 1
        summary.update(
            {
                "num_trajectories": len(self.trajectories),
                "stage_oob_rate": stage_oob / max(checked, 1),
                "rho_oob_rate": rho_oob / max(checked, 1),
            }
        )
        return summary

    def initial_worker_states(self, count: int, seed: int) -> list[TrajectoryWorkerState]:
        rng = np.random.default_rng(seed)
        return [self.random_worker_state(rng) for _ in range(count)]

    def random_worker_state(self, rng: np.random.Generator) -> TrajectoryWorkerState:
        demo_idx = int(rng.integers(0, len(self.trajectories)))
        return TrajectoryWorkerState(
            demo_idx=demo_idx,
            obs_index=0,
            previous_chunk=np.zeros((self.horizon, self.action_dim), dtype=np.float32),
        )

    def chunk_at(self, state: TrajectoryWorkerState) -> dict:
        traj = self.trajectories[state.demo_idx]
        max_start = max(0, len(traj.actions) - self.horizon - 1)
        t = int(np.clip(state.obs_index, 0, max_start))
        stage, rho = _find_stage_and_rho(t, traj.stage_boundaries)
        obs_array = traj.state if self.obs_full_state else traj.ee_pos
        prev_action = (
            state.previous_chunk[-1].copy()
            if state.previous_chunk.size
            else np.zeros((self.action_dim,), dtype=np.float32)
        )
        return {
            "obs": obs_array[t].copy(),
            "action_chunk": traj.actions[t : t + self.horizon].copy(),
            "ee_pos_seq": traj.ee_pos[t : t + self.horizon].copy(),
            "prev_action": prev_action,
            "stage": np.int64(stage),
            "rho_start": np.float32(rho),
            "task_name": traj.task_name,
            "demo_idx": np.int64(state.demo_idx),
            "obs_index": np.int64(t),
        }

    def advance_worker_state(
        self,
        state: TrajectoryWorkerState,
        predicted_chunk: np.ndarray,
        *,
        min_advance: int,
        max_advance: int,
        matcher: str = "artifact_anchor",
        match_cost: str = "mse",
    ) -> tuple[TrajectoryWorkerState, dict[str, float]]:
        traj = self.trajectories[state.demo_idx]
        t0 = int(state.obs_index)
        old_stage, old_rho = _find_stage_and_rho(t0, traj.stage_boundaries)
        start, end = traj.stage_boundaries[old_stage]
        lower = min(max(t0 + max(min_advance, 1), int(start)), len(traj.ee_pos) - 1)
        upper = min(max(t0 + max_advance, lower), int(end) - 1, len(traj.ee_pos) - 1)
        if lower >= upper:
            next_state = self.random_worker_state(np.random.default_rng(t0 + state.demo_idx + 1))
            return next_state, {
                "rho_advance": 1.0,
                "stall": 0.0,
                "reset": 1.0,
                "matcher_cost": 0.0,
                "matcher_used": 0.0,
            }

        pred_path = traj.ee_pos[t0] + np.cumsum(predicted_chunk[:, :3].astype(np.float32), axis=0)
        terminal = pred_path[-1]
        matcher_cost = 0.0
        matcher_used = 0.0

        if matcher == "artifact_anchor":
            next_index, matcher_cost = self._artifact_anchor_next_index(
                traj=traj,
                stage=old_stage,
                old_rho=old_rho,
                terminal=terminal,
                lower=lower,
                upper=upper,
                match_cost=match_cost,
            )
            matcher_used = 1.0
        elif matcher == "artifact_anchor_path_mse":
            next_index, matcher_cost = self._artifact_path_next_index(
                traj=traj,
                stage=old_stage,
                old_rho=old_rho,
                pred_path=pred_path,
                lower=lower,
                upper=upper,
                cloud_mode="anchor",
            )
            matcher_used = 1.0
        elif matcher == "artifact_cloud_path_mse":
            next_index, matcher_cost = self._artifact_path_next_index(
                traj=traj,
                stage=old_stage,
                old_rho=old_rho,
                pred_path=pred_path,
                lower=lower,
                upper=upper,
                cloud_mode="cloud_nearest",
            )
            matcher_used = 1.0
        elif matcher == "demo_nearest":
            next_index, matcher_cost = self._demo_nearest_next_index(
                traj, terminal, lower, upper, match_cost=match_cost
            )
        else:
            raise ValueError(f"Unknown dynamic matcher: {matcher}")

        new_stage, new_rho = _find_stage_and_rho(next_index, traj.stage_boundaries)
        if new_stage > old_stage:
            rho_advance = 1.0 - old_rho + new_rho
        else:
            rho_advance = new_rho - old_rho
        done = next_index >= len(traj.actions) - self.horizon - 1 or next_index >= int(end) - 1
        if done:
            next_state = self.random_worker_state(np.random.default_rng(next_index + state.demo_idx + 17))
            reset = 1.0
        else:
            next_state = TrajectoryWorkerState(
                demo_idx=state.demo_idx,
                obs_index=next_index,
                previous_chunk=predicted_chunk.astype(np.float32, copy=True),
            )
            reset = 0.0
        return next_state, {
            "rho_advance": float(max(rho_advance, 0.0)),
            "stall": float(next_index <= t0),
            "reset": reset,
            "matcher_cost": float(matcher_cost),
            "matcher_used": matcher_used,
        }

    def _demo_nearest_next_index(
        self,
        traj: RolloutTrajectory,
        terminal: np.ndarray,
        lower: int,
        upper: int,
        match_cost: str = "mse",
    ) -> tuple[int, float]:
        candidates = traj.ee_pos[lower : upper + 1]
        costs = _point_match_cost(candidates, terminal, match_cost)
        nearest_offset = int(np.argmin(costs))
        return int(lower + nearest_offset), float(costs[nearest_offset])

    def _artifact_anchor_next_index(
        self,
        *,
        traj: RolloutTrajectory,
        stage: int,
        old_rho: float,
        terminal: np.ndarray,
        lower: int,
        upper: int,
        match_cost: str,
    ) -> tuple[int, float]:
        artifact = traj.artifact
        if stage < 0 or stage >= artifact.num_stages:
            return self._demo_nearest_next_index(traj, terminal, lower, upper, match_cost=match_cost)

        start, end = traj.stage_boundaries[stage]
        width = max(int(end) - int(start), 1)
        lower_rho = max(old_rho, (lower - int(start)) / width)
        upper_rho = min(1.0, (upper - int(start)) / width)
        phase_mask = (artifact.phase_grid >= lower_rho) & (artifact.phase_grid <= upper_rho)
        phase_indices = np.flatnonzero(phase_mask)
        if phase_indices.size == 0:
            return self._demo_nearest_next_index(traj, terminal, lower, upper, match_cost=match_cost)

        anchors = artifact.anchor[stage, phase_indices, :]
        costs = _point_match_cost(anchors, terminal, match_cost)
        phase_idx = int(phase_indices[int(np.argmin(costs))])
        rho_next = float(artifact.phase_grid[phase_idx])
        next_index = int(round(int(start) + rho_next * width))
        next_index = int(np.clip(next_index, lower, upper))
        return next_index, float(np.min(costs))

    def _artifact_path_next_index(
        self,
        *,
        traj: RolloutTrajectory,
        stage: int,
        old_rho: float,
        pred_path: np.ndarray,
        lower: int,
        upper: int,
        cloud_mode: str,
    ) -> tuple[int, float]:
        artifact = traj.artifact
        if stage < 0 or stage >= artifact.num_stages:
            return self._demo_nearest_next_index(traj, pred_path[-1], lower, upper, match_cost="mse")

        start, end = traj.stage_boundaries[stage]
        width = max(int(end) - int(start), 1)
        lower_rho = max(old_rho, (lower - int(start)) / width)
        upper_rho = min(1.0, (upper - int(start)) / width)
        phase_indices = np.flatnonzero((artifact.phase_grid >= lower_rho) & (artifact.phase_grid <= upper_rho))
        if phase_indices.size == 0:
            return self._demo_nearest_next_index(traj, pred_path[-1], lower, upper, match_cost="mse")

        horizon = pred_path.shape[0]
        candidate_costs = []
        candidate_indices = []
        for phase_idx in phase_indices:
            phase_seq = np.linspace(phase_idx, artifact.num_phase - 1, horizon)
            phase_seq = np.clip(np.round(phase_seq).astype(np.int32), phase_idx, artifact.num_phase - 1)
            if cloud_mode == "anchor":
                ref_path = artifact.anchor[stage, phase_seq, :]
                cost = float(np.mean((pred_path - ref_path) ** 2))
            elif cloud_mode == "cloud_nearest":
                cloud = artifact.phase_points[stage, :, phase_seq, :]  # [N, H, 3]
                per_demo = np.mean((cloud - pred_path[None, :, :]) ** 2, axis=(1, 2))
                cost = float(np.min(per_demo))
            else:
                raise ValueError(f"Unknown cloud_mode: {cloud_mode}")
            candidate_costs.append(cost)
            candidate_indices.append(int(phase_idx))

        best_offset = int(np.argmin(np.asarray(candidate_costs)))
        best_phase = candidate_indices[best_offset]
        rho_next = float(artifact.phase_grid[best_phase])
        next_index = int(round(int(start) + rho_next * width))
        next_index = int(np.clip(next_index, lower, upper))
        return next_index, float(candidate_costs[best_offset])


def _point_match_cost(candidates: np.ndarray, point: np.ndarray, match_cost: str) -> np.ndarray:
    diff = candidates - point[None, :]
    if match_cost == "mse":
        return np.mean(diff**2, axis=-1)
    if match_cost == "l2":
        return np.linalg.norm(diff, axis=-1)
    raise ValueError(f"Unknown match_cost: {match_cost}")


def _find_artifact(hdf5_path: Path, artifact_dir: Path) -> Path | None:
    """Find the matching artifact .npz for a given HDF5 task file.

    Tries exact name match first, then suffix-stripped variants.
    """
    stem = hdf5_path.stem

    # Direct match: <stem>_patcs.npz
    candidate = artifact_dir / f"{stem}_patcs.npz"
    if candidate.exists():
        return candidate

    # Suite parent match: <artifact_dir>/<suite>/<stem>_patcs.npz.
    candidate = artifact_dir / hdf5_path.parent.name / f"{stem}_patcs.npz"
    if candidate.exists():
        return candidate

    # The artifact might have been built from a different HDF5 naming convention.
    # Try dropping trailing _demo suffix.
    import re
    stem2 = re.sub(r"_demo$", "", stem)
    candidate2 = artifact_dir / f"{stem2}_patcs.npz"
    if candidate2.exists():
        return candidate2

    candidate2 = artifact_dir / hdf5_path.parent.name / f"{stem2}_patcs.npz"
    if candidate2.exists():
        return candidate2

    return None


def _find_artifact_for_task_name(task_name: str, task_dir: Path, artifact_dir: Path) -> Path | None:
    """Find a rollout artifact for a grouped task dir.

    Supports flat smoke names such as ``libero_spatial_task_02_patcs.npz`` and
    nested names such as ``libero_spatial/task_02_patcs.npz``.
    """

    suite = task_dir.parent.name
    task = task_dir.name
    candidates = [
        artifact_dir / f"{task_name}_patcs.npz",
        artifact_dir / suite / f"{task}_patcs.npz",
        artifact_dir / suite / f"{task_name}_patcs.npz",
        artifact_dir / f"{task}_patcs.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
