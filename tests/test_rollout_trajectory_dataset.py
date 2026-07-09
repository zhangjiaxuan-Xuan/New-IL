from __future__ import annotations

from pathlib import Path

import numpy as np

from new_il.data.patcs_artifacts import PatcsArtifactConfig, build_patcs_artifact_from_rollouts
from new_il.training.dataset import RolloutTrajectoryDataset


def _write_rollout(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    state = np.stack(
        [
            t,
            offset + t,
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.full_like(t, 0.1),
            np.full_like(t, 0.2),
        ],
        axis=-1,
    )
    actions = np.zeros((16, 7), dtype=np.float32)
    actions[:, 0] = 1.0 / 15.0
    actions[:, 1] = 1.0 / 15.0
    actions[:8, -1] = -1.0
    actions[8:, -1] = 1.0
    np.savez_compressed(
        path,
        language=np.asarray("move along the diagonal"),
        actions=actions,
        success=np.asarray(True),
        metadata_json=np.asarray(
            '{"schema_version":1,"task_suite_name":"libero_spatial",'
            '"task_idx":2,"episode_idx":0,"round_idx":0,"seed":7}'
        ),
        **{"observation.state": state},
    )


def test_rollout_trajectory_dataset_advances_from_predicted_chunk(tmp_path: Path) -> None:
    task_dir = tmp_path / "by_task" / "libero_spatial" / "task_02"
    for idx, offset in enumerate([0.0, 0.01, -0.01, 0.02]):
        _write_rollout(task_dir / "success_rollouts" / f"demo_{idx}.npz", offset)

    artifact_dir = tmp_path / "artifacts"
    build_patcs_artifact_from_rollouts(
        task_dir,
        artifact_dir / "libero_spatial_task_02_patcs.npz",
        PatcsArtifactConfig(num_demos=4, num_phase=8, obs_key="ee_pos", event_radius=0.02),
    )

    dataset = RolloutTrajectoryDataset(task_dir, artifact_dir, horizon=4, obs_full_state=True, verbose=False)
    health = dataset.health_summary()
    assert health["num_trajectories"] == 4
    assert health["all_zero_rate"] == 0.0
    assert health["repeated_action_rate"] < 1.0
    assert health["xyz_norm_mean"] > 0.0
    assert health["stage_oob_rate"] == 0.0
    assert health["rho_oob_rate"] == 0.0

    worker = dataset.initial_worker_states(1, seed=1)[0]
    batch = dataset.chunk_at(worker)
    next_worker, stats = dataset.advance_worker_state(
        worker,
        batch["action_chunk"],
        min_advance=1,
        max_advance=4,
        matcher="artifact_anchor",
        match_cost="mse",
    )

    assert next_worker.obs_index > worker.obs_index
    assert stats["rho_advance"] > 0.0
    assert stats["stall"] == 0.0
    assert stats["matcher_used"] == 1.0
    assert stats["matcher_cost"] >= 0.0

    next_worker_demo, demo_stats = dataset.advance_worker_state(
        worker,
        batch["action_chunk"],
        min_advance=1,
        max_advance=4,
        matcher="demo_nearest",
        match_cost="l2",
    )
    assert next_worker_demo.obs_index > worker.obs_index
    assert demo_stats["matcher_used"] == 0.0

    for matcher in ("artifact_anchor_path_mse", "artifact_cloud_path_mse"):
        next_worker_path, path_stats = dataset.advance_worker_state(
            worker,
            batch["action_chunk"],
            min_advance=1,
            max_advance=4,
            matcher=matcher,
        )
        assert next_worker_path.obs_index > worker.obs_index
        assert path_stats["rho_advance"] > 0.0
        assert path_stats["matcher_used"] == 1.0
        assert path_stats["matcher_cost"] >= 0.0
