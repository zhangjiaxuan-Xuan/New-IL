from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from new_il.data.lerobot_to_hdf5 import convert_lerobot_dataset


def test_convert_lerobot_dataset_writes_hdf5(tmp_path) -> None:
    dataset_root = tmp_path / "lerobot"
    data_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    tasks = pd.DataFrame({"task_index": [0, 1]}, index=["pick object", "place object"])
    tasks.to_parquet(meta_dir / "tasks.parquet")

    rows = []
    for episode_index, task_index in [(0, 0), (1, 0), (2, 1)]:
        for frame_index in range(4):
            rows.append(
                {
                    "observation.state": np.arange(8, dtype=np.float32) + frame_index,
                    "action": np.full((7,), frame_index, dtype=np.float32),
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "task_index": task_index,
                }
            )
    pd.DataFrame(rows).to_parquet(data_dir / "file-000.parquet")

    written = convert_lerobot_dataset(
        dataset_root,
        tmp_path / "hdf5",
        max_tasks=1,
        max_demos_per_task=2,
    )

    assert set(written) == {0}
    hdf5_path = written[0]
    with h5py.File(hdf5_path, "r") as handle:
        assert handle.attrs["num_demos"] == 2
        assert handle.attrs["task_name"] == "pick object"
        assert sorted(handle["data"].keys()) == ["demo_0", "demo_1"]
        demo = handle["data"]["demo_0"]
        assert demo["obs"]["ee_pos"].shape == (4, 3)
        assert demo["obs"]["ee_states"].shape == (4, 8)
        assert demo["obs"]["joint_state"].shape == (4, 7)
        assert demo["actions"].shape == (4, 7)
