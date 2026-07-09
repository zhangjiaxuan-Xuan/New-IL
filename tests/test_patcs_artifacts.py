import numpy as np
import h5py

from new_il.data.patcs_artifacts import (
    PatcsArtifactConfig,
    build_patcs_artifact,
    build_patcs_artifact_from_rollouts,
)


def _write_demo(group: h5py.Group, name: str, offset: float) -> None:
    demo = group.create_group(name)
    obs = demo.create_group("obs")
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    obs.create_dataset(
        "ee_pos",
        data=np.stack([t, offset + t**2, np.zeros_like(t)], axis=-1),
    )
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:4, -1] = -1.0
    actions[4:, -1] = 1.0
    demo.create_dataset("actions", data=actions)


def test_build_patcs_artifact_writes_npz_and_manifest(tmp_path) -> None:
    hdf5_path = tmp_path / "task.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        data = handle.create_group("data")
        _write_demo(data, "demo_0", 0.0)
        _write_demo(data, "demo_1", 0.1)
        _write_demo(data, "demo_2", -0.1)
        _write_demo(data, "demo_3", 0.05)

    output = tmp_path / "artifact.npz"
    build_patcs_artifact(
        hdf5_path,
        output,
        PatcsArtifactConfig(num_demos=4, num_phase=6, margin=0.01),
    )

    assert output.exists()
    assert output.with_suffix(".json").exists()
    artifact = np.load(output)
    assert int(artifact["schema_version"]) == 1
    assert artifact["phase_points"].shape == (2, 4, 6, 3)
    assert artifact["anchor_points"].shape == (2, 6, 3)
    assert artifact["event_mask"].shape == (2, 6)
    assert artifact["event_mask"][:, 0].all()
    assert artifact["event_mask"][:, -1].all()
    assert artifact["hull_equations"].shape[-1] == 4
    assert artifact["hull_equation_counts"].shape == (2, 6)
    assert artifact["stage_boundaries"].shape == (4, 2, 2)
    assert artifact["transition_counts"].tolist() == [1, 1, 1, 1]


def _write_rollout(path, offset: float, success: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    state = np.stack(
        [
            t,
            offset + t**2,
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.full_like(t, 0.1),
            np.full_like(t, 0.2),
        ],
        axis=-1,
    )
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:4, -1] = -1.0
    actions[4:, -1] = 1.0
    np.savez_compressed(
        path,
        language=np.asarray("pick up the cup"),
        actions=actions,
        success=np.asarray(success),
        **{"observation.state": state},
    )


def test_build_patcs_artifact_from_rollout_npz(tmp_path) -> None:
    rollout_dir = tmp_path / "rollouts"
    _write_rollout(rollout_dir / "success_rollouts" / "demo_0.npz", 0.0)
    _write_rollout(rollout_dir / "success_rollouts" / "demo_1.npz", 0.1)
    _write_rollout(rollout_dir / "success_rollouts" / "demo_2.npz", -0.1)
    _write_rollout(rollout_dir / "failed_rollouts" / "demo_bad.npz", 0.3, success=False)

    output = tmp_path / "rollout_patcs.npz"
    build_patcs_artifact_from_rollouts(
        rollout_dir,
        output,
        PatcsArtifactConfig(num_demos=3, num_phase=6, obs_key="ee_pos"),
    )

    artifact = np.load(output)
    assert artifact["phase_points"].shape == (2, 3, 6, 3)
    assert artifact["stage_boundaries"].shape == (3, 2, 2)
    assert artifact["demo_ids"].shape == (3,)
