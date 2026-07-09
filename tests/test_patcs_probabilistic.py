import numpy as np

from new_il.patcs_probabilistic import (
    ProbabilisticTubeConfig,
    black_hole_event_scale,
    build_probabilistic_trajectory_tube,
    gaussian_tube_nll,
    normal_plane_mahalanobis,
    olive_segment_scale,
    precision_channel_scale,
    tube_radius_diagnostics,
    tube_surface_mesh,
)


def _demo_set() -> list[np.ndarray]:
    phase = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    demos = []
    for offset in (-0.02, 0.0, 0.015, 0.03):
        xyz = np.stack(
            [
                0.1 * phase + offset,
                np.sin(phase * np.pi) * 0.08 + offset * 0.2,
                0.2 + np.cos(phase * np.pi) * 0.04,
            ],
            axis=-1,
        )
        demos.append(xyz.astype(np.float32))
    return demos


def test_black_hole_event_scale_contracts_to_zero_at_transition() -> None:
    scale, mask = black_hole_event_scale(21, np.array([10]), event_window=5, contraction_power=2.0)
    assert scale[10] == 0.0
    assert scale[5] == 1.0
    assert scale[8] < scale[7]
    assert scale[9] < scale[8]
    assert mask[10]


def test_olive_segment_scale_is_wide_in_middle_and_narrow_at_boundaries() -> None:
    scale = olive_segment_scale(21, np.array([10]), olive_power=0.75)
    assert scale[0] < 1e-6
    assert scale[10] < 1e-6
    assert scale[20] < 1e-6
    assert scale[5] > scale[2]
    assert scale[5] > scale[8]
    assert scale[15] > scale[12]
    assert scale[15] > scale[18]


def test_precision_channel_scale_creates_longer_corridor_than_blackhole() -> None:
    scale = precision_channel_scale(
        41,
        np.array([20]),
        precision_channel_window=8,
        precision_channel_radius=0.01,
        min_envelope_std=0.04,
    )
    assert scale[20] < scale[16] < scale[12]
    assert scale[20] < scale[24] < scale[28]
    assert scale[11] == 1.0
    assert scale[29] == 1.0


def test_probabilistic_tube_shapes_and_event_collapse() -> None:
    tube = build_probabilistic_trajectory_tube(
        _demo_set(),
        target_index=1,
        transitions=np.array([10, 30]),
        config=ProbabilisticTubeConfig(event_window=4, event_radius=0.003),
    )
    assert tube.aligned_points.shape == (4, 40, 3)
    assert tube.mean.shape == (40, 3)
    assert tube.frame.shape == (40, 3, 3)
    assert tube.cov2.shape == (40, 2, 2)
    assert tube.base_cov2.shape == (40, 2, 2)
    assert tube.olive_scale.shape == (40,)
    assert tube.channel_scale.shape == (40,)
    assert tube.blackhole_mask.shape == (40,)
    np.testing.assert_allclose(tube.mean[10], tube.target_xyz[10])
    np.testing.assert_allclose(tube.mean[30], tube.target_xyz[30])
    for cov in tube.cov2:
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_probabilistic_tube_prefers_inside_points() -> None:
    tube = build_probabilistic_trajectory_tube(
        _demo_set(),
        target_index=1,
        transitions=np.array([10, 30]),
        config=ProbabilisticTubeConfig(event_window=4, event_radius=0.003, min_std=0.006),
    )
    t = 20
    inside = tube.mean[t]
    outside = tube.mean[t] + tube.frame[t, 1] * 0.2
    assert normal_plane_mahalanobis(inside, tube, t) < normal_plane_mahalanobis(outside, tube, t)
    assert gaussian_tube_nll(inside, tube, t) < gaussian_tube_nll(outside, tube, t)


def test_tube_surface_mesh_shapes() -> None:
    tube = build_probabilistic_trajectory_tube(
        _demo_set(),
        target_index=1,
        transitions=np.array([10, 30]),
        config=ProbabilisticTubeConfig(surface_sides=12),
    )
    vertices, faces, density = tube_surface_mesh(tube, stride=4)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert density.ndim == 1
    assert len(vertices) == len(density) * 12


def test_tube_radius_diagnostics_flags_no_suspicious_minima_on_smooth_demo() -> None:
    tube = build_probabilistic_trajectory_tube(
        _demo_set(),
        target_index=1,
        transitions=np.array([10, 30]),
        config=ProbabilisticTubeConfig(
            event_window=4,
            blackhole_window=1,
            min_envelope_std=0.015,
            radius_smooth_window=5,
        ),
    )
    diag = tube_radius_diagnostics(tube)
    assert set(diag) == {
        "radius_minor",
        "radius_major",
        "radius_area",
        "cov_det",
        "nearest_transition_distance",
        "event_scale",
        "olive_scale",
        "channel_scale",
        "blackhole_mask",
        "suspicious_non_event_minima",
    }
    assert not bool(np.any(diag["suspicious_non_event_minima"]))
    assert diag["radius_area"][5] > diag["radius_area"][10]
    assert diag["radius_area"][20] > diag["radius_area"][10]


def test_global_3d_gaussian_mode_builds_valid_tube() -> None:
    tube = build_probabilistic_trajectory_tube(
        _demo_set(),
        target_index=1,
        transitions=np.array([10, 30]),
        config=ProbabilisticTubeConfig(
            density_mode="global_3d_gaussian",
            event_window=4,
            precision_channel_window=8,
            precision_channel_radius=0.01,
        ),
    )
    diag = tube_radius_diagnostics(tube)
    assert tube.cov2.shape == (40, 2, 2)
    assert np.all(np.linalg.eigvalsh(tube.cov2) > 0.0)
    assert diag["channel_scale"][10] < diag["channel_scale"][5]
    assert diag["radius_area"][10] < diag["radius_area"][5]
