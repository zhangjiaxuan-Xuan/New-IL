from __future__ import annotations

import numpy as np

from new_il.libero.rollout import _format_lerobot_observation, _save_trajectory_npz, action_health_summary


def _raw_obs() -> dict:
    return {
        "agentview_image": np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3),
        "robot0_eye_in_hand_image": np.ones((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }


def test_format_lerobot_observation_matches_mem_schema() -> None:
    formatted = _format_lerobot_observation(_raw_obs())
    assert set(formatted) == {"pixels", "agent_pos"}
    assert formatted["pixels"]["image"].shape == (4, 4, 3)
    assert formatted["pixels"]["image2"].shape == (4, 4, 3)
    assert formatted["pixels"]["image"].flags.c_contiguous
    assert formatted["agent_pos"].shape == (8,)
    np.testing.assert_allclose(formatted["agent_pos"][:3], [1.0, 2.0, 3.0])


def test_save_trajectory_npz_writes_mem_compatible_keys(tmp_path) -> None:
    formatted = _format_lerobot_observation(_raw_obs())
    output = tmp_path / "traj.npz"
    _save_trajectory_npz(
        {
            "language": "pick up the cup",
            "actions": [np.ones((7,), dtype=np.float32), np.full((7,), 2.0, dtype=np.float32)],
            "images": [formatted["pixels"]["image"], formatted["pixels"]["image"]],
            "images2": [formatted["pixels"]["image2"], formatted["pixels"]["image2"]],
            "state": [formatted["agent_pos"], formatted["agent_pos"]],
            "success": True,
            "metadata": {"task_idx": 0},
        },
        output,
    )
    data = np.load(output)
    assert data["actions"].shape == (2, 7)
    assert data["observation.images.image"].shape == (2, 4, 4, 3)
    assert data["observation.images.image2"].shape == (2, 4, 4, 3)
    assert data["observation.state"].shape == (2, 8)
    assert data["text"].shape == (2, 8)
    assert data["vision"].shape == (2, 8)
    assert bool(data["success"])


def test_action_health_flags_zero_actions_unstable() -> None:
    health = action_health_summary([np.zeros((7,), dtype=np.float32)])
    assert health["all_zero"] is True
    assert health["stable"] is False
