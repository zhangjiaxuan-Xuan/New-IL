import h5py
import numpy as np

from new_il.data.patcs_artifacts import PatcsArtifactConfig, build_patcs_artifact


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
