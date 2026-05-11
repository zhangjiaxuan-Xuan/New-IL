import numpy as np

from new_il.patcs import (
    EventChannelConfig,
    SmoothLossConfig,
    TubeLossConfig,
    anchor_transition_contraction,
    build_olive_trajectory_cloud,
    dtw_progress_match,
    event_channel_loss,
    interpolate_phase_trajectory,
    inter_chunk_smoothness_loss,
    intra_chunk_smoothness_loss,
    next_observation_index_from_progress,
    olive_tube_loss,
    resample_segment,
)


def test_dtw_progress_match_tracks_reference_step() -> None:
    reference = np.stack(
        [
            np.linspace(0.0, 1.0, 10),
            np.zeros(10),
        ],
        axis=-1,
    ).astype(np.float32)
    query = reference[:6] + np.array([0.01, 0.0], dtype=np.float32)

    match = dtw_progress_match(query, [reference])

    assert match.reference_index == 0
    assert 4 <= match.reference_step <= 6
    assert 0.4 <= match.rho <= 0.7
    assert next_observation_index_from_progress(match, horizon=2, reference_length=len(reference)) <= 8


def test_olive_cloud_contracts_at_gripper_events() -> None:
    phase = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    demos = []
    for offset in [-0.03, 0.0, 0.03]:
        demos.append(
            np.stack(
                [
                    phase,
                    offset * np.sin(np.pi * phase),
                    np.zeros_like(phase),
                ],
                axis=-1,
            )
        )
    cloud = build_olive_trajectory_cloud(np.stack(demos, axis=0), event_radius=0.01, interior_radius=0.2)

    start_radius = float(np.mean(cloud.radius[0]))
    mid_radius = float(np.mean(cloud.radius[len(phase) // 2]))
    end_radius = float(np.mean(cloud.radius[-1]))

    assert mid_radius > start_radius * 10
    assert mid_radius > end_radius * 10
    assert cloud.contraction[0, 0] == 0.0
    assert cloud.contraction[-1, 0] == 0.0


def test_anchor_transition_contraction_creates_olive_between_anchor_nodes() -> None:
    phase = np.linspace(0.0, 1.0, 11, dtype=np.float32)

    contraction = anchor_transition_contraction(phase, transition_rhos=(0.0, 1.0))

    assert contraction[0] == 0.0
    assert contraction[-1] == 0.0
    assert contraction[5] == np.max(contraction)


def test_anchor_reward_prefers_original_path_inside_cloud() -> None:
    phase = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    demos = np.stack(
        [
            np.stack([phase, np.zeros_like(phase), np.zeros_like(phase)], axis=-1),
            np.stack([phase, 0.08 * np.sin(np.pi * phase), np.zeros_like(phase)], axis=-1),
        ],
        axis=0,
    )
    cloud = build_olive_trajectory_cloud(demos, event_radius=0.01, interior_radius=0.2)
    anchor = demos[0, 4:8]
    near_anchor = anchor.copy()
    away_inside_cloud = anchor.copy()
    away_inside_cloud[:, 1] += 0.08
    config = TubeLossConfig(v_max=0.05, delta=0.2, anchor_reward_weight=0.1, anchor_sigma=0.1)

    assert olive_tube_loss(near_anchor, cloud, rho_start=0.2, anchor=anchor, config=config) < olive_tube_loss(
        away_inside_cloud, cloud, rho_start=0.2, anchor=anchor, config=config
    )


def test_olive_distance_uses_anchor_not_nearest_demo_point() -> None:
    phase = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    anchor = np.stack([phase, np.zeros_like(phase), np.zeros_like(phase)], axis=-1)
    other_demo = np.stack([phase, np.ones_like(phase), np.zeros_like(phase)], axis=-1)
    cloud = build_olive_trajectory_cloud(
        np.stack([anchor, other_demo], axis=0),
        anchor_index=0,
        event_radius=0.1,
        interior_radius=0.2,
        empirical_scale=0.0,
    )

    predicted_near_other_demo = other_demo[3:4]
    predicted_near_anchor = anchor[3:4]
    config = TubeLossConfig(v_max=0.0, delta=0.01, anchor_reward_weight=0.0, temperature=0.1)

    assert olive_tube_loss(
        predicted_near_anchor,
        cloud,
        rho_start=0.5,
        config=config,
    ) < olive_tube_loss(
        predicted_near_other_demo,
        cloud,
        rho_start=0.5,
        config=config,
    )


def test_action_resample_can_avoid_event_points() -> None:
    segment = np.array(
        [
            [0.0, 0.0],
            [10.0, 1.0],
            [11.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    resampled = resample_segment(segment, 6, avoid_event_points=True)

    np.testing.assert_allclose(resampled[0], segment[0])
    np.testing.assert_allclose(resampled[-1], segment[-1])
    assert np.all(resampled[1:-1, 0] >= 10.0)


def test_interpolate_phase_trajectory_matches_prediction_length_not_raw_time() -> None:
    phase_points = np.stack(
        [
            np.linspace(0.0, 1.0, 9),
            np.zeros(9),
        ],
        axis=-1,
    ).astype(np.float32)
    rho_values = np.array([0.10, 0.20, 0.45, 0.90], dtype=np.float32)

    anchor = interpolate_phase_trajectory(phase_points, rho_values)

    assert anchor.shape == (4, 2)
    np.testing.assert_allclose(anchor[:, 0], rho_values, atol=1e-6)


def test_event_channel_strongly_binds_anchor_transition() -> None:
    phase = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    anchor = np.stack([phase, np.zeros_like(phase), np.zeros_like(phase)], axis=-1)
    other_demo = anchor.copy()
    other_demo[:, 1] = 0.05 * np.sin(np.pi * phase)
    cloud = build_olive_trajectory_cloud(
        np.stack([anchor, other_demo], axis=0),
        anchor_index=0,
        event_radius=0.01,
        interior_radius=0.2,
    )
    on_event = anchor[-1:]
    off_event = on_event + np.array([[0.0, 0.05, 0.0]], dtype=np.float32)
    tube_config = TubeLossConfig(v_min=0.0, v_max=0.0, delta=0.01)
    event_config = EventChannelConfig(radius=0.01, weight=10.0, event_rhos=(1.0,))

    assert event_channel_loss(
        off_event,
        cloud,
        rho_start=1.0,
        tube_config=tube_config,
        event_config=event_config,
    ) > event_channel_loss(
        on_event,
        cloud,
        rho_start=1.0,
        tube_config=tube_config,
        event_config=event_config,
    )


def test_event_channel_only_triggers_when_progress_window_hits_event() -> None:
    phase = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    anchor = np.stack([phase, np.zeros_like(phase), np.zeros_like(phase)], axis=-1)
    cloud = build_olive_trajectory_cloud(np.stack([anchor, anchor], axis=0))
    predicted = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)

    assert event_channel_loss(
        predicted,
        cloud,
        rho_start=0.5,
        tube_config=TubeLossConfig(v_min=0.0, v_max=0.0, delta=0.01),
        event_config=EventChannelConfig(event_rhos=(1.0,)),
    ) == 0.0


def test_intra_chunk_smoothness_penalizes_jitter() -> None:
    smooth = np.stack([np.linspace(0.0, 1.0, 8), np.zeros(8)], axis=-1).astype(np.float32)
    jitter = smooth.copy()
    jitter[::2, 1] = 1.0

    assert intra_chunk_smoothness_loss(jitter) > intra_chunk_smoothness_loss(smooth)


def test_inter_chunk_smoothness_penalizes_boundary_jump() -> None:
    previous = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    continuous = np.array([[1.5], [2.0], [2.5]], dtype=np.float32)
    jump = np.array([[5.0], [5.5], [6.0]], dtype=np.float32)
    config = SmoothLossConfig(boundary_position_weight=1.0, boundary_velocity_weight=1.0)

    assert inter_chunk_smoothness_loss(previous, jump, config) > inter_chunk_smoothness_loss(
        previous,
        continuous,
        config,
    )
