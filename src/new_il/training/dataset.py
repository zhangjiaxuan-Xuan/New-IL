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


@dataclass(frozen=True)
class ChunkItem:
    obs: np.ndarray           # [obs_dim]
    action_chunk: np.ndarray  # [H, 7]
    ee_pos_seq: np.ndarray    # [H, 3]  GT positions for PATCS
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
                yield ChunkItem(
                    obs=obs_array[t].copy(),
                    action_chunk=actions[t : t + horizon].copy(),
                    ee_pos_seq=ee_pos[t : t + horizon].copy(),
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
