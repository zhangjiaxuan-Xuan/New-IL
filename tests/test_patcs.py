import numpy as np

from new_il.patcs import (
    TubeLossConfig,
    build_phase_cloud,
    gripper_transition_indices,
    progress_backward_rate,
    tube_loss,
    tube_violation_rate,
)


def test_gripper_transition_indices() -> None:
    actions = np.array(
        [
            [0.0, -1.0],
            [0.1, -1.0],
            [0.2, 1.0],
            [0.3, 1.0],
            [0.4, -1.0],
        ],
        dtype=np.float32,
    )

    assert gripper_transition_indices(actions) == [2, 4]


def test_phase_cloud_and_tube_loss_prefers_in_tube_prediction() -> None:
    demo_a = np.array([[0.0, -1.0], [0.5, -1.0], [1.0, 1.0], [1.5, 1.0]], dtype=np.float32)
    demo_b = np.array([[0.1, -1.0], [0.4, -1.0], [1.1, 1.0], [1.4, 1.0]], dtype=np.float32)
    cloud = build_phase_cloud([demo_a, demo_b], num_phase=8)
    first_stage = cloud[0]

    in_tube = np.array([[0.05], [0.45]], dtype=np.float32)
    out_of_tube = np.array([[3.0], [3.5]], dtype=np.float32)
    config = TubeLossConfig(sigma=0.2, gamma=3.0, v_max=0.5, delta=0.2)

    assert tube_loss(in_tube, first_stage, rho_start=0.0, config=config) < tube_loss(
        out_of_tube, first_stage, rho_start=0.0, config=config
    )
    assert tube_violation_rate(in_tube, first_stage, rho_start=0.0, config=config) == 0.0


def test_progress_backward_rate() -> None:
    assert progress_backward_rate(np.array([0.0, 0.2, 0.1, 0.4], dtype=np.float32)) == 1 / 3
