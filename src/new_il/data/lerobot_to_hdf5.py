"""Convert LeRobot LIBERO parquet datasets to New-IL per-task HDF5 files.

The converter reads only state/action metadata columns, so it does not decode
image payloads. Output layout matches ``new_il.data.patcs_artifacts`` and
``new_il.training.dataset``:

    data/
      demo_0/
        obs/
          ee_pos      [T, 3] float32
          ee_states   [T, 8] float32
          joint_state [T, 7] float32
        actions       [T, 7] float32

Episodes are grouped by ``task_index`` and written as one HDF5 file per task.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from tqdm import tqdm


DATA_COLUMNS = [
    "observation.state",
    "action",
    "episode_index",
    "frame_index",
    "task_index",
]


@dataclass
class Episode:
    episode_index: int
    task_index: int
    states: np.ndarray
    actions: np.ndarray


def _safe_task_name(task: str, task_index: int) -> str:
    name = task.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return f"task_{task_index:03d}_{name or 'unnamed'}"


def _load_task_names(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to read LeRobot parquet metadata.") from exc

    tasks = pd.read_parquet(tasks_path)
    mapping: dict[int, str] = {}
    for task_name, row in tasks.iterrows():
        mapping[int(row["task_index"])] = str(task_name)
    return mapping


def _iter_parquet_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not files:
        files = sorted((dataset_root / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return files


def _iter_episodes(dataset_root: Path) -> Iterable[Episode]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to read LeRobot parquet data.") from exc

    for parquet_path in tqdm(_iter_parquet_files(dataset_root), desc="parquet"):
        frame = pd.read_parquet(parquet_path, columns=DATA_COLUMNS)
        for episode_index, episode in frame.groupby("episode_index", sort=True):
            episode = episode.sort_values("frame_index")
            task_values = episode["task_index"].unique()
            if len(task_values) != 1:
                raise ValueError(f"{parquet_path}: episode {episode_index} spans multiple tasks")
            states = np.stack(episode["observation.state"].to_numpy()).astype(np.float32)
            actions = np.stack(episode["action"].to_numpy()).astype(np.float32)
            if states.ndim != 2 or states.shape[1] < 3:
                raise ValueError(f"{parquet_path}: bad state shape {states.shape}")
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"{parquet_path}: bad action shape {actions.shape}")
            yield Episode(
                episode_index=int(episode_index),
                task_index=int(task_values[0]),
                states=states,
                actions=actions,
            )


def _write_task_hdf5(output_path: Path, task_name: str, episodes: list[Episode]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        data_group = handle.create_group("data")
        for demo_index, episode in enumerate(sorted(episodes, key=lambda ep: ep.episode_index)):
            demo_group = data_group.create_group(f"demo_{demo_index}")
            obs_group = demo_group.create_group("obs")
            ee_states = episode.states.astype(np.float32, copy=False)
            obs_group.create_dataset("ee_pos", data=ee_states[:, :3], compression="gzip")
            obs_group.create_dataset("ee_states", data=ee_states, compression="gzip")
            obs_group.create_dataset("joint_state", data=ee_states[:, :7], compression="gzip")
            demo_group.create_dataset(
                "actions",
                data=episode.actions.astype(np.float32, copy=False),
                compression="gzip",
            )
        handle.attrs["num_demos"] = len(episodes)
        handle.attrs["task_name"] = task_name


def convert_lerobot_dataset(
    dataset_root: Path,
    output_dir: Path,
    max_tasks: int | None = None,
    max_demos_per_task: int | None = None,
) -> dict[int, Path]:
    """Convert a local LeRobot dataset checkout to per-task HDF5 files."""

    dataset_root = dataset_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_names = _load_task_names(dataset_root)

    selected_tasks: set[int] | None = None
    grouped: dict[int, list[Episode]] = defaultdict(list)
    for episode in _iter_episodes(dataset_root):
        if selected_tasks is None:
            selected_tasks = set()
        if episode.task_index not in selected_tasks:
            if max_tasks is not None and len(selected_tasks) >= max_tasks:
                continue
            selected_tasks.add(episode.task_index)
        if episode.task_index not in selected_tasks:
            continue
        if (
            max_demos_per_task is not None
            and len(grouped[episode.task_index]) >= max_demos_per_task
        ):
            continue
        grouped[episode.task_index].append(episode)

    if not grouped:
        raise RuntimeError(f"No episodes converted from {dataset_root}")

    written: dict[int, Path] = {}
    manifest_tasks = []
    for task_index in sorted(grouped):
        task_name = task_names.get(task_index, f"task {task_index}")
        safe_name = _safe_task_name(task_name, task_index)
        output_path = output_dir / f"{safe_name}.hdf5"
        _write_task_hdf5(output_path, task_name, grouped[task_index])
        written[task_index] = output_path
        manifest_tasks.append(
            {
                "task_index": task_index,
                "task": task_name,
                "hdf5": str(output_path),
                "num_demos": len(grouped[task_index]),
            }
        )
        print(f"{task_index:03d}: {len(grouped[task_index])} demos -> {output_path}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "max_tasks": max_tasks,
        "max_demos_per_task": max_demos_per_task,
        "tasks": manifest_tasks,
    }
    (output_dir / "new_il_lerobot_hdf5_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a local LeRobot LIBERO parquet dataset to New-IL HDF5."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/L202500340/data/huggingface/lerobot/HuggingFaceVLA/libero"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/L202500340/data/libero_lerobot_hdf5"),
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-demos-per-task", type=int, default=None)
    args = parser.parse_args()

    convert_lerobot_dataset(
        args.dataset_root,
        args.output,
        max_tasks=args.max_tasks,
        max_demos_per_task=args.max_demos_per_task,
    )


if __name__ == "__main__":
    main()
