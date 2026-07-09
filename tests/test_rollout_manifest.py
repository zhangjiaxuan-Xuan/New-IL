from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from new_il.data.rollout_manifest import build_rollout_manifest


def _write_rollout(path: Path, *, suite: str, task: int, success: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(0.01, 0.05, 5, dtype=np.float32)
    actions[:, -1] = 1.0
    state = np.zeros((5, 8), dtype=np.float32)
    state[:, 0] = np.linspace(0.0, 0.1, 5, dtype=np.float32)
    np.savez_compressed(
        path,
        language=np.asarray("pick up the cup"),
        actions=actions,
        success=np.asarray(success),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_suite_name": suite,
                    "task_idx": task,
                    "episode_idx": 0,
                    "round_idx": 0,
                    "seed": 7,
                }
            )
        ),
        **{
            "observation.state": state,
            "observation.images.image": np.zeros((5, 4, 4, 3), dtype=np.uint8),
            "observation.images.image2": np.zeros((5, 4, 4, 3), dtype=np.uint8),
        },
    )


def test_build_rollout_manifest_groups_success_and_reports_bad(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_rollout(raw / "worker_00" / "success_rollouts" / "a.npz", suite="libero_goal", task=3)
    _write_rollout(
        raw / "worker_00" / "failed_rollouts" / "b.npz",
        suite="libero_goal",
        task=3,
        success=False,
    )
    bad = raw / "worker_01" / "success_rollouts" / "bad.npz"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not a zip", encoding="utf-8")

    out = tmp_path / "manifest"
    summary = build_rollout_manifest(raw, out)

    assert summary["total_npz_readable"] == 2
    assert summary["total_bad_npz"] == 1
    assert summary["success"] == 1
    assert summary["failed"] == 1
    group = summary["groups"]["libero_goal/task_03"]
    assert group["success"] == 1
    assert group["failed"] == 1

    linked = list((out / "by_task" / "libero_goal" / "task_03" / "success_rollouts").glob("*.npz"))
    assert len(linked) == 1
    assert linked[0].is_symlink()
    assert (out / "manifest.json").exists()
    assert (out / "summary.json").exists()
