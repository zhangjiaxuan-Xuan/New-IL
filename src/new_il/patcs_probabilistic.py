from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ProbabilisticTubeConfig:
    density_mode: str = "phase_2d"
    event_window: int = 6
    blackhole_window: int = 2
    precision_channel_window: int = 12
    precision_channel_radius: float = 0.018
    event_radius: float = 0.006
    min_std: float = 0.008
    min_envelope_std: float = 0.018
    covariance_shrinkage: float = 0.15
    contraction_power: float = 2.0
    olive_power: float = 0.75
    radius_smooth_window: int = 9
    max_radius_ratio: float = 1.2
    target_mean_blend: float = 0.35
    iso_sigma: float = 2.0
    surface_sides: int = 24


@dataclass(frozen=True)
class ProbabilisticTrajectoryTube:
    aligned_points: Array       # [N, T, 3]
    target_xyz: Array           # [T, 3]
    mean: Array                 # [T, 3]
    frame: Array                # [T, 3, 3], rows are tangent, normal_a, normal_b
    cov2: Array                 # [T, 2, 2], contracted normal-plane covariance
    base_cov2: Array            # [T, 2, 2], pre-contraction normal-plane covariance
    olive_scale: Array          # [T]
    channel_scale: Array        # [T]
    event_scale: Array          # [T]
    event_mask: Array           # [T] bool
    blackhole_mask: Array       # [T] bool
    transitions: Array          # [K]
    config: ProbabilisticTubeConfig


def resample_trajectory(values: Array, target_len: int) -> Array:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must be [T, D], got {values.shape}")
    if target_len < 2:
        raise ValueError(f"target_len must be >= 2, got {target_len}")
    if len(values) == target_len:
        return values.copy()
    src_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.stack([np.interp(dst_x, src_x, values[:, d]) for d in range(values.shape[1])], axis=-1).astype(np.float32)


def align_trajectories_to_target(demos_xyz: list[Array] | Array, target_len: int) -> Array:
    if isinstance(demos_xyz, np.ndarray) and demos_xyz.ndim == 3 and demos_xyz.shape[1] == target_len:
        return demos_xyz.astype(np.float32, copy=True)
    return np.stack([resample_trajectory(np.asarray(demo), target_len) for demo in demos_xyz], axis=0)


def _normalize(vector: Array, fallback: Array) -> Array:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return fallback.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def trajectory_local_frames(target_xyz: Array) -> Array:
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError(f"target_xyz must be [T, 3], got {target_xyz.shape}")
    frames = np.zeros((target_xyz.shape[0], 3, 3), dtype=np.float32)
    global_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    prev_tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for t in range(target_xyz.shape[0]):
        if t == 0:
            tangent_raw = target_xyz[min(1, len(target_xyz) - 1)] - target_xyz[0]
        elif t == len(target_xyz) - 1:
            tangent_raw = target_xyz[-1] - target_xyz[-2]
        else:
            tangent_raw = target_xyz[t + 1] - target_xyz[t - 1]
        tangent = _normalize(tangent_raw, prev_tangent)
        prev_tangent = tangent
        up = alt_up if abs(float(np.dot(tangent, global_up))) > 0.92 else global_up
        normal_a = _normalize(np.cross(tangent, up), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        normal_b = _normalize(np.cross(tangent, normal_a), np.array([0.0, 0.0, 1.0], dtype=np.float32))
        frames[t, 0] = tangent
        frames[t, 1] = normal_a
        frames[t, 2] = normal_b
    return frames


def black_hole_event_scale(
    length: int,
    transitions: Array,
    *,
    event_window: int,
    contraction_power: float,
) -> tuple[Array, Array]:
    if length < 1:
        raise ValueError("length must be positive")
    transitions = np.asarray(transitions, dtype=np.int32)
    scale = np.ones((length,), dtype=np.float32)
    event_mask = np.zeros((length,), dtype=bool)
    if len(transitions) == 0:
        return scale, event_mask
    window = max(int(event_window), 1)
    power = max(float(contraction_power), 1.0)
    for transition in transitions:
        center = int(np.clip(transition, 0, length - 1))
        lo = max(0, center - window)
        hi = min(length, center + window + 1)
        for t in range(lo, hi):
            u = abs(t - center) / float(window)
            scale[t] = min(scale[t], float(np.clip(u, 0.0, 1.0) ** power))
            event_mask[t] = True
    return scale, event_mask


def olive_segment_scale(
    length: int,
    transitions: Array,
    *,
    olive_power: float,
) -> Array:
    if length < 1:
        raise ValueError("length must be positive")
    transitions = np.asarray(transitions, dtype=np.int32)
    boundaries = [0]
    boundaries.extend(int(np.clip(t, 0, length - 1)) for t in transitions)
    boundaries.append(length - 1)
    boundaries = sorted(set(boundaries))
    scale = np.ones((length,), dtype=np.float32)
    power = max(float(olive_power), 0.1)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        denom = max(float(right - left), 1.0)
        for t in range(left, right + 1):
            s = (t - left) / denom
            scale[t] = min(scale[t], float(np.sin(np.pi * s) ** power))
    return np.clip(scale, 0.0, 1.0).astype(np.float32, copy=False)


def blackhole_mask(
    length: int,
    transitions: Array,
    *,
    blackhole_window: int,
) -> Array:
    mask = np.zeros((length,), dtype=bool)
    window = max(int(blackhole_window), 0)
    for transition in np.asarray(transitions, dtype=np.int32):
        center = int(np.clip(transition, 0, length - 1))
        lo = max(0, center - window)
        hi = min(length, center + window + 1)
        mask[lo:hi] = True
    return mask


def precision_channel_scale(
    length: int,
    transitions: Array,
    *,
    precision_channel_window: int,
    precision_channel_radius: float,
    min_envelope_std: float,
) -> Array:
    scale = np.ones((length,), dtype=np.float32)
    window = max(int(precision_channel_window), 1)
    floor = float(np.clip(precision_channel_radius / max(min_envelope_std, 1e-8), 0.05, 1.0))
    for transition in np.asarray(transitions, dtype=np.int32):
        center = int(np.clip(transition, 0, length - 1))
        lo = max(0, center - window)
        hi = min(length, center + window + 1)
        for t in range(lo, hi):
            u = abs(t - center) / float(window)
            # Smoothstep gives a long navigable corridor and avoids a cliff before the final anchor.
            smooth = u * u * (3.0 - 2.0 * u)
            scale[t] = min(scale[t], float(floor + (1.0 - floor) * smooth))
    return scale


def _regularized_cov2(points2: Array, *, min_std: float, shrinkage: float) -> Array:
    points2 = np.asarray(points2, dtype=np.float32)
    if points2.shape[0] <= 1:
        return np.eye(2, dtype=np.float32) * float(min_std ** 2)
    cov = np.cov(points2, rowvar=False, bias=False).astype(np.float32)
    if cov.shape == ():
        cov = np.eye(2, dtype=np.float32) * float(cov)
    diag_mean = float(np.trace(cov) / 2.0)
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    cov = (1.0 - shrink) * cov + shrink * np.eye(2, dtype=np.float32) * max(diag_mean, min_std ** 2)
    cov += np.eye(2, dtype=np.float32) * float(min_std ** 2)
    return cov.astype(np.float32, copy=False)


def _smooth_1d(values: Array, window: int) -> Array:
    values = np.asarray(values, dtype=np.float32)
    window = int(window)
    if window <= 1 or len(values) <= 2:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)


def _limit_radius_slope(radii: Array, *, max_ratio: float) -> Array:
    radii = np.asarray(radii, dtype=np.float32).copy()
    ratio = max(float(max_ratio), 1.01)
    for i in range(1, len(radii)):
        radii[i] = max(radii[i], radii[i - 1] / ratio)
    for i in range(len(radii) - 2, -1, -1):
        radii[i] = max(radii[i], radii[i + 1] / ratio)
    return radii


def _cov2_with_target_radii(cov: Array, target_radii: Array) -> Array:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    target = np.maximum(np.asarray(target_radii, dtype=np.float32), 1e-8)
    return (vecs @ np.diag(target ** 2) @ vecs.T).astype(np.float32, copy=False)


def _global_3d_covariance(aligned: Array, mean: Array, *, min_std: float, shrinkage: float) -> Array:
    offsets = (aligned - mean[None, :, :]).reshape(-1, 3)
    if offsets.shape[0] <= 1:
        cov = np.eye(3, dtype=np.float32) * float(min_std ** 2)
    else:
        cov = np.cov(offsets, rowvar=False, bias=False).astype(np.float32)
    diag_mean = float(np.trace(cov) / 3.0)
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    cov = (1.0 - shrink) * cov + shrink * np.eye(3, dtype=np.float32) * max(diag_mean, min_std ** 2)
    cov += np.eye(3, dtype=np.float32) * float(min_std ** 2)
    return cov.astype(np.float32, copy=False)


def _project_global_cov_to_frame(cov3: Array, frame: Array) -> Array:
    normals = frame[1:]
    return (normals @ cov3 @ normals.T).astype(np.float32, copy=False)


def build_probabilistic_trajectory_tube(
    demos_xyz: list[Array] | Array,
    *,
    target_index: int,
    transitions: Array,
    config: ProbabilisticTubeConfig = ProbabilisticTubeConfig(),
) -> ProbabilisticTrajectoryTube:
    if isinstance(demos_xyz, np.ndarray):
        if demos_xyz.ndim != 3 or demos_xyz.shape[-1] != 3:
            raise ValueError(f"demos_xyz must be [N, T, 3] or list of [T, 3], got {demos_xyz.shape}")
        if not 0 <= target_index < demos_xyz.shape[0]:
            raise ValueError(f"target_index {target_index} out of range")
        target_xyz = np.asarray(demos_xyz[target_index], dtype=np.float32)
        aligned = align_trajectories_to_target(demos_xyz, len(target_xyz))
    else:
        if not 0 <= target_index < len(demos_xyz):
            raise ValueError(f"target_index {target_index} out of range")
        target_xyz = np.asarray(demos_xyz[target_index], dtype=np.float32)
        aligned = align_trajectories_to_target(demos_xyz, len(target_xyz))

    frame = trajectory_local_frames(target_xyz)
    event_scale, event_mask = black_hole_event_scale(
        len(target_xyz),
        transitions,
        event_window=config.event_window,
        contraction_power=config.contraction_power,
    )
    olive_scale = olive_segment_scale(
        len(target_xyz),
        transitions,
        olive_power=config.olive_power,
    )
    channel_scale = precision_channel_scale(
        len(target_xyz),
        transitions,
        precision_channel_window=config.precision_channel_window,
        precision_channel_radius=config.precision_channel_radius,
        min_envelope_std=config.min_envelope_std,
    )
    precise_mask = blackhole_mask(
        len(target_xyz),
        transitions,
        blackhole_window=config.blackhole_window,
    )
    cloud_mean = aligned.mean(axis=0).astype(np.float32, copy=False)
    blend = float(np.clip(config.target_mean_blend, 0.0, 1.0))
    mean = ((1.0 - blend) * cloud_mean + blend * target_xyz).astype(np.float32, copy=False)
    cov2 = np.zeros((len(target_xyz), 2, 2), dtype=np.float32)
    base_cov2 = np.zeros_like(cov2)
    transitions_set = set(int(x) for x in np.asarray(transitions, dtype=np.int32).tolist())
    if config.density_mode not in {"phase_2d", "global_3d_gaussian"}:
        raise ValueError(f"Unknown density_mode: {config.density_mode}")
    global_cov3 = _global_3d_covariance(
        aligned,
        mean,
        min_std=config.min_std,
        shrinkage=config.covariance_shrinkage,
    )
    for t in range(len(target_xyz)):
        if config.density_mode == "global_3d_gaussian":
            base = _project_global_cov_to_frame(global_cov3, frame[t])
            base += np.eye(2, dtype=np.float32) * float(config.min_std ** 2)
        else:
            offsets = aligned[:, t, :] - mean[t]
            normal_offsets = offsets @ frame[t, 1:].T
            base = _regularized_cov2(
                normal_offsets,
                min_std=config.min_std,
                shrinkage=config.covariance_shrinkage,
            )
        base_cov2[t] = base

    base_vals = np.linalg.eigvalsh(base_cov2)
    base_radii = np.sqrt(np.maximum(base_vals, 1e-12))
    smoothed_radii = np.zeros_like(base_radii)
    for dim in range(2):
        radius = np.maximum(base_radii[:, dim], float(config.min_envelope_std))
        radius = _smooth_1d(radius, config.radius_smooth_window)
        radius = _limit_radius_slope(radius, max_ratio=config.max_radius_ratio)
        smoothed_radii[:, dim] = radius

    for t in range(len(target_xyz)):
        base_cov2[t] = _cov2_with_target_radii(base_cov2[t], smoothed_radii[t])
        scale = float(event_scale[t])
        olive = float(olive_scale[t])
        channel = float(channel_scale[t])
        envelope_scale = olive * channel
        envelope_scale = max(envelope_scale, float(config.event_radius / max(config.min_envelope_std, 1e-8)))
        if precise_mask[t]:
            envelope_scale = min(envelope_scale, scale)
            mean[t] = target_xyz[t]
        cov2[t] = base_cov2[t] * (envelope_scale ** 2) + np.eye(2, dtype=np.float32) * float(config.event_radius ** 2)
        if t in transitions_set:
            mean[t] = target_xyz[t]
            cov2[t] = np.eye(2, dtype=np.float32) * float(config.event_radius ** 2)
    return ProbabilisticTrajectoryTube(
        aligned_points=aligned.astype(np.float32, copy=False),
        target_xyz=target_xyz.astype(np.float32, copy=False),
        mean=mean.astype(np.float32, copy=False),
        frame=frame,
        cov2=cov2,
        base_cov2=base_cov2,
        olive_scale=olive_scale,
        channel_scale=channel_scale,
        event_scale=event_scale,
        event_mask=event_mask,
        blackhole_mask=precise_mask,
        transitions=np.asarray(transitions, dtype=np.int32),
        config=config,
    )


def tube_radius_diagnostics(tube: ProbabilisticTrajectoryTube) -> dict[str, Array]:
    vals = np.linalg.eigvalsh(tube.cov2)
    radii = np.sqrt(np.maximum(vals, 0.0)).astype(np.float32, copy=False)
    cov_det = np.linalg.det(tube.cov2).astype(np.float32, copy=False)
    nearest = np.full((len(tube.target_xyz),), 999, dtype=np.int32)
    transitions = np.asarray(tube.transitions, dtype=np.int32)
    for t in range(len(nearest)):
        if len(transitions):
            nearest[t] = int(np.min(np.abs(transitions - t)))
    radius_area = np.sqrt(np.maximum(cov_det, 0.0)).astype(np.float32, copy=False)
    local_min = np.zeros_like(radius_area, dtype=bool)
    for t in range(1, len(radius_area) - 1):
        neighbor_floor = min(float(radius_area[t - 1]), float(radius_area[t + 1]))
        prominent = float(radius_area[t]) < 0.9 * neighbor_floor
        local_min[t] = radius_area[t] < radius_area[t - 1] and radius_area[t] < radius_area[t + 1] and prominent
    suspicious = local_min & (~tube.blackhole_mask) & (nearest > tube.config.event_window)
    return {
        "radius_minor": radii[:, 0],
        "radius_major": radii[:, 1],
        "radius_area": radius_area,
        "cov_det": cov_det,
        "nearest_transition_distance": nearest,
        "event_scale": tube.event_scale,
        "olive_scale": tube.olive_scale,
        "channel_scale": tube.channel_scale,
        "blackhole_mask": tube.blackhole_mask,
        "suspicious_non_event_minima": suspicious,
    }


def normal_plane_mahalanobis(point_xyz: Array, tube: ProbabilisticTrajectoryTube, t: int) -> float:
    point = np.asarray(point_xyz, dtype=np.float32)
    delta = point - tube.mean[t]
    uv = delta @ tube.frame[t, 1:].T
    inv = np.linalg.inv(tube.cov2[t])
    return float(uv.T @ inv @ uv)


def gaussian_tube_nll(point_xyz: Array, tube: ProbabilisticTrajectoryTube, t: int) -> float:
    point = np.asarray(point_xyz, dtype=np.float32)
    delta = point - tube.mean[t]
    uv = delta @ tube.frame[t, 1:].T
    cov = tube.cov2[t]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError(f"cov2 at t={t} is not positive definite")
    inv = np.linalg.inv(cov)
    return float(0.5 * (uv.T @ inv @ uv + logdet + 2.0 * np.log(2.0 * np.pi)))


def tube_surface_mesh(
    tube: ProbabilisticTrajectoryTube,
    *,
    stride: int = 2,
    iso_sigma: float | None = None,
    sides: int | None = None,
) -> tuple[Array, Array, Array]:
    stride = max(int(stride), 1)
    iso = float(tube.config.iso_sigma if iso_sigma is None else iso_sigma)
    side_count = max(int(tube.config.surface_sides if sides is None else sides), 8)
    indices = np.arange(0, len(tube.target_xyz), stride, dtype=np.int32)
    if indices[-1] != len(tube.target_xyz) - 1:
        indices = np.concatenate([indices, np.array([len(tube.target_xyz) - 1], dtype=np.int32)])
    angles = np.linspace(0.0, 2.0 * np.pi, side_count, endpoint=False, dtype=np.float32)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    vertices = np.zeros((len(indices), side_count, 3), dtype=np.float32)
    density = np.zeros((len(indices),), dtype=np.float32)
    for row, t in enumerate(indices):
        vals, vecs = np.linalg.eigh(tube.cov2[int(t)])
        vals = np.maximum(vals, 1e-12)
        ellipse2 = (unit @ vecs.T) * (np.sqrt(vals)[None, :] * iso)
        vertices[row] = tube.mean[int(t)] + ellipse2 @ tube.frame[int(t), 1:]
        density[row] = float(1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(tube.cov2[int(t)]))))
    faces = []
    for row in range(len(indices) - 1):
        for side in range(side_count):
            a = row * side_count + side
            b = row * side_count + ((side + 1) % side_count)
            c = (row + 1) * side_count + ((side + 1) % side_count)
            d = (row + 1) * side_count + side
            faces.append([a, b, c])
            faces.append([a, c, d])
    return vertices.reshape(-1, 3), np.asarray(faces, dtype=np.int32), density
