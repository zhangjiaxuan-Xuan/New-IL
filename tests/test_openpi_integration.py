from __future__ import annotations

import numpy as np

from new_il.integrations.openpi import (
    OpenPILiberoConfig,
    build_openpi_libero_payload,
    libero_state_for_openpi,
    openpi_libero_image,
    quat_to_axis_angle,
)


def test_quat_to_axis_angle_identity() -> None:
    np.testing.assert_allclose(quat_to_axis_angle(np.array([0, 0, 0, 1], dtype=np.float32)), np.zeros(3))


def test_openpi_libero_image_rotates_and_pads() -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    processed = openpi_libero_image(image, size=4)
    assert processed.shape == (4, 4, 3)
    assert processed.dtype == np.uint8
    # The source aspect ratio is preserved, so the rotated red pixel lands in
    # the lower-right corner of the non-padded image band.
    assert processed[2, 3, 0] == 255


def test_build_openpi_libero_payload_schema() -> None:
    obs = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.float32),
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }
    payload = build_openpi_libero_payload(obs, "pick up the cup", OpenPILiberoConfig(resize_size=224))
    assert set(payload) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert payload["observation/image"].shape == (224, 224, 3)
    assert payload["observation/wrist_image"].dtype == np.uint8
    assert payload["observation/state"].shape == (8,)
    np.testing.assert_allclose(libero_state_for_openpi(obs), payload["observation/state"])
    assert payload["prompt"] == "pick up the cup"
