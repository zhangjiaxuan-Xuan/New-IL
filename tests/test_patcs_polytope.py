import numpy as np

from new_il.patcs import (
    TubeLossConfig,
    build_polytope_trajectory_cloud,
    interpolated_polytope_distance,
    polytope_section_distance,
    polytope_tube_loss,
)


def _box_points(z: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [-1.0, -1.0, z],
            [-1.0, 1.0, z],
            [1.0, -1.0, z],
            [1.0, 1.0, z],
            [0.0, 0.0, z + 0.2],
        ],
        dtype=np.float32,
    )


def test_polytope_cloud_contains_demo_points_with_margin() -> None:
    phase_points = np.stack([_box_points(0.0), _box_points(0.5), _box_points(1.0)], axis=1)
    cloud = build_polytope_trajectory_cloud(
        phase_points,
        transition_rhos=(0.0, 1.0),
        margin=0.05,
    )

    middle = cloud.sections[1]
    assert not middle.is_event
    for point in phase_points[:, 1, :]:
        assert polytope_section_distance(point, middle, event_radius=cloud.event_radius) == 0.0


def test_polytope_event_sections_collapse_to_anchor_point() -> None:
    phase_points = np.stack([_box_points(0.0), _box_points(0.5), _box_points(1.0)], axis=1)
    cloud = build_polytope_trajectory_cloud(
        phase_points,
        anchor_index=0,
        transition_rhos=(0.0, 1.0),
        event_radius=1e-4,
    )

    assert cloud.sections[0].is_event
    assert polytope_section_distance(phase_points[0, 0], cloud.sections[0], event_radius=cloud.event_radius) == 0.0
    assert polytope_section_distance(phase_points[1, 0], cloud.sections[0], event_radius=cloud.event_radius) > 1000.0


def test_interpolated_polytope_distance_is_zero_inside_neighboring_clouds() -> None:
    phase_points = np.stack([_box_points(0.0), _box_points(0.5), _box_points(1.0)], axis=1)
    cloud = build_polytope_trajectory_cloud(
        phase_points,
        transition_rhos=(0.0, 1.0),
        margin=0.1,
    )

    assert interpolated_polytope_distance(np.array([0.0, 0.0, 0.5], dtype=np.float32), cloud, 0.5) == 0.0


def test_polytope_tube_loss_penalizes_outside_more_than_inside() -> None:
    phase_points = np.stack([_box_points(0.0), _box_points(0.5), _box_points(1.0)], axis=1)
    cloud = build_polytope_trajectory_cloud(
        phase_points,
        transition_rhos=(0.0, 1.0),
        margin=0.05,
    )
    inside = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    outside = np.array([[3.0, 3.0, 0.5]], dtype=np.float32)
    config = TubeLossConfig(v_min=0.0, v_max=0.0, delta=0.01)

    assert polytope_tube_loss(outside, cloud, rho_start=0.5, config=config) > polytope_tube_loss(
        inside,
        cloud,
        rho_start=0.5,
        config=config,
    )
