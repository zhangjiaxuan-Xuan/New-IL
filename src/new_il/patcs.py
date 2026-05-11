from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TubeLossConfig:
    """Numerical knobs for the PA-TCS tube objective."""

    sigma: float = 0.05
    gamma: float = 1.0
    temperature: float = 0.25
    v_min: float = 0.0
    v_max: float = 1.0
    delta: float = 0.05
    dt: float = 1.0
    anchor_reward_weight: float = 0.05
    anchor_sigma: float = 0.05


@dataclass(frozen=True)
class EventChannelConfig:
    """Strong node constraint for anchor gripper-transition points."""

    radius: float = 0.01
    weight: float = 10.0
    event_rhos: tuple[float, ...] = (0.0, 1.0)


@dataclass(frozen=True)
class SmoothLossConfig:
    """Smoothness weights for action chunks and chunk boundaries."""

    velocity_weight: float = 1.0
    acceleration_weight: float = 0.25
    boundary_position_weight: float = 1.0
    boundary_velocity_weight: float = 0.25


@dataclass(frozen=True)
class DtwProgress:
    """Progress match from a generated prefix into a reference demonstration."""

    reference_index: int
    reference_step: int
    rho: float
    cost: float
    path: list[tuple[int, int]]


@dataclass(frozen=True)
class OliveTrajectoryCloud:
    """Task-level phase cloud contracted by anchor transition channels."""

    points: Array
    center: Array
    anchor: Array
    base_radius: Array
    contraction: Array
    radius: Array
    phase: Array
    event_radius: float
    anchor_index: int


@dataclass(frozen=True)
class HullSection:
    """One phase cross-section of an irregular trajectory cloud."""

    phase: float
    points: Array
    anchor: Array
    equations: Array | None
    margin: float
    is_event: bool


@dataclass(frozen=True)
class PolytopeTrajectoryCloud:
    """Discrete phase cloud made of irregular hull cross-sections."""

    sections: tuple[HullSection, ...]
    phase: Array
    event_rhos: tuple[float, ...]
    event_radius: float
    margin: float


def gripper_transition_indices(actions: Array, gripper_index: int = -1, threshold: float = 0.0) -> list[int]:
    """Return indices where the binarized gripper state changes."""

    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must be [T, D], got {actions.shape}.")
    gripper = actions[:, gripper_index] > threshold
    changes = np.flatnonzero(gripper[1:] != gripper[:-1]) + 1
    return changes.astype(int).tolist()


def segment_boundaries(length: int, transition_indices: list[int]) -> list[tuple[int, int]]:
    """Build inclusive-exclusive stage boundaries from event indices."""

    if length < 2:
        raise ValueError("length must be at least 2.")
    points = [0, *sorted(i for i in transition_indices if 0 < i < length), length]
    return [(start, end) for start, end in zip(points[:-1], points[1:]) if end > start]


def normalized_phase(length: int) -> Array:
    """Evenly spaced phase coordinates in [0, 1]."""

    if length < 1:
        raise ValueError("length must be positive.")
    if length == 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def resample_segment(segment: Array, num_phase: int, *, avoid_event_points: bool = False) -> Array:
    """Resample a trajectory segment to a fixed phase grid.

    For action trajectories, `avoid_event_points=True` keeps the two gripper
    transition endpoints exact and interpolates only the interior. This blurs
    time inside the phase without smearing the event constraints themselves.
    """

    segment = np.asarray(segment, dtype=np.float32)
    if segment.ndim != 2:
        raise ValueError(f"segment must be [T, D], got {segment.shape}.")
    if avoid_event_points and segment.shape[0] > 2 and num_phase > 2:
        interior = segment[1:-1]
        source_phase = normalized_phase(interior.shape[0])
        target_phase = normalized_phase(num_phase - 2)
        dims = [np.interp(target_phase, source_phase, interior[:, d]) for d in range(segment.shape[1])]
        resampled = np.empty((num_phase, segment.shape[1]), dtype=np.float32)
        resampled[0] = segment[0]
        resampled[-1] = segment[-1]
        resampled[1:-1] = np.stack(dims, axis=-1)
        return resampled
    source_phase = normalized_phase(segment.shape[0])
    target_phase = normalized_phase(num_phase)
    dims = [np.interp(target_phase, source_phase, segment[:, d]) for d in range(segment.shape[1])]
    return np.stack(dims, axis=-1).astype(np.float32, copy=False)


def dtw_alignment_path(
    query: Array,
    reference: Array,
    *,
    open_end: bool = False,
) -> tuple[list[tuple[int, int]], float]:
    """Align a generated/executed prefix to a reference trajectory with DTW."""

    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError(f"query/reference must be [T, D], got {query.shape} and {reference.shape}.")
    if query.shape[-1] != reference.shape[-1]:
        raise ValueError("query and reference must share feature dimension.")

    q_len, r_len = query.shape[0], reference.shape[0]
    costs = np.full((q_len + 1, r_len + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    local = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=-1)
    for i in range(1, q_len + 1):
        for j in range(1, r_len + 1):
            costs[i, j] = local[i - 1, j - 1] + min(
                costs[i - 1, j],
                costs[i, j - 1],
                costs[i - 1, j - 1],
            )

    end_j = int(np.argmin(costs[q_len, 1:]) + 1) if open_end else r_len
    i, j = q_len, end_j
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = (
            (costs[i - 1, j - 1], i - 1, j - 1),
            (costs[i - 1, j], i - 1, j),
            (costs[i, j - 1], i, j - 1),
        )
        _, i, j = min(choices, key=lambda item: item[0])
    path.reverse()
    return path, float(costs[q_len, end_j] / max(q_len + end_j, 1))


def dtw_progress_match(query: Array, references: list[Array]) -> DtwProgress:
    """Find the training trajectory progress that corresponds to the latest generated point.

    The returned `reference_step` is the observation index to use when selecting
    the next conditioning observation O from the matched demonstration.
    """

    if not references:
        raise ValueError("at least one reference trajectory is required.")
    best: DtwProgress | None = None
    last_query_index = np.asarray(query).shape[0] - 1
    for reference_index, reference in enumerate(references):
        reference = np.asarray(reference, dtype=np.float32)
        path, cost = dtw_alignment_path(query, reference, open_end=True)
        matched_steps = [ref_step for query_step, ref_step in path if query_step == last_query_index]
        reference_step = int(round(float(np.mean(matched_steps)))) if matched_steps else path[-1][1]
        rho = reference_step / max(reference.shape[0] - 1, 1)
        candidate = DtwProgress(
            reference_index=reference_index,
            reference_step=reference_step,
            rho=float(rho),
            cost=cost,
            path=path,
        )
        if best is None or candidate.cost < best.cost:
            best = candidate
    assert best is not None
    return best


def next_observation_index_from_progress(match: DtwProgress, horizon: int, reference_length: int) -> int:
    """Map DTW progress to the next demonstration observation index for O conditioning."""

    if reference_length <= 0:
        raise ValueError("reference_length must be positive.")
    return int(np.clip(match.reference_step + horizon, 0, reference_length - 1))


def build_phase_cloud(
    demonstrations: list[Array],
    *,
    num_phase: int = 32,
    gripper_index: int = -1,
    threshold: float = 0.0,
    continuous_dims: slice | list[int] | Array = slice(None, -1),
    avoid_event_points: bool = False,
) -> Array:
    """Construct a phase-indexed trajectory cloud from demonstrations.

    The returned array is shaped [R, N, P, D], where R is the number of stages,
    N the number of demonstrations, P the phase grid size, and D the continuous
    action dimension count.
    """

    if not demonstrations:
        raise ValueError("at least one demonstration is required.")

    all_segments: list[list[Array]] = []
    expected_stages: int | None = None
    for demo in demonstrations:
        demo = np.asarray(demo, dtype=np.float32)
        transitions = gripper_transition_indices(demo, gripper_index=gripper_index, threshold=threshold)
        segments = segment_boundaries(len(demo), transitions)
        if expected_stages is None:
            expected_stages = len(segments)
        elif len(segments) != expected_stages:
            raise ValueError(
                "all demonstrations must expose the same number of gripper-defined stages "
                f"for this smoke implementation; got {len(segments)} and expected {expected_stages}."
            )
        all_segments.append([demo[start:end, continuous_dims] for start, end in segments])

    stages = []
    for stage_idx in range(expected_stages or 0):
        stage_cloud = [
            resample_segment(
                demo_segments[stage_idx],
                num_phase,
                avoid_event_points=avoid_event_points,
            )
            for demo_segments in all_segments
        ]
        stages.append(np.stack(stage_cloud, axis=0))
    return np.stack(stages, axis=0).astype(np.float32, copy=False)


def build_olive_trajectory_cloud(
    phase_points: Array,
    *,
    anchor_index: int = 0,
    event_radius: float = 0.02,
    interior_radius: float = 0.15,
    empirical_scale: float = 1.0,
    olive_power: float = 1.0,
    transition_rhos: tuple[float, ...] = (0.0, 1.0),
) -> OliveTrajectoryCloud:
    """Build a task-level cloud, then contract it around anchor transition channels.

    Data construction order:
    1. use a same-length anchor trajectory as the retained task-time skeleton;
    2. estimate base cloud thickness from same-task demonstrations;
    3. contract that cloud at anchor state-transition phases. The olive shape is
       therefore the result of anchor transition channels, not a separate prior.
    """

    points = np.asarray(phase_points, dtype=np.float32)
    if points.ndim != 3:
        raise ValueError(f"phase_points must be [N, P, D], got {points.shape}.")
    if not 0 <= anchor_index < points.shape[0]:
        raise ValueError(f"anchor_index must be in [0, {points.shape[0]}), got {anchor_index}.")
    phase = normalized_phase(points.shape[1])
    center = points.mean(axis=0)
    empirical = points.std(axis=0)
    base_radius = interior_radius + empirical_scale * empirical
    contraction = anchor_transition_contraction(
        phase,
        transition_rhos=transition_rhos,
        olive_power=olive_power,
    ).reshape(-1, 1)
    radius = event_radius + contraction * base_radius
    return OliveTrajectoryCloud(
        points=points,
        center=center.astype(np.float32, copy=False),
        anchor=points[anchor_index].astype(np.float32, copy=False),
        base_radius=base_radius.astype(np.float32, copy=False),
        contraction=contraction.astype(np.float32, copy=False),
        radius=radius.astype(np.float32, copy=False),
        phase=phase,
        event_radius=event_radius,
        anchor_index=anchor_index,
    )


def anchor_transition_contraction(
    phase: Array,
    *,
    transition_rhos: tuple[float, ...] = (0.0, 1.0),
    olive_power: float = 1.0,
) -> Array:
    """Return [0, 1] cloud expansion caused by distance from anchor transitions."""

    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim != 1:
        raise ValueError(f"phase must be [P], got {phase.shape}.")
    transitions = np.asarray(transition_rhos, dtype=np.float32)
    if transitions.ndim != 1 or transitions.size == 0:
        raise ValueError("transition_rhos must contain at least one phase value.")
    transitions = np.clip(np.sort(transitions), 0.0, 1.0)
    contraction = np.ones_like(phase, dtype=np.float32)
    for left, right in zip(transitions[:-1], transitions[1:]):
        mask = (phase >= left) & (phase <= right)
        width = max(float(right - left), 1e-8)
        local = (phase[mask] - left) / width
        contraction[mask] = np.sin(np.pi * local) ** olive_power
    for transition in transitions:
        contraction[np.isclose(phase, transition)] = 0.0
    return contraction.astype(np.float32, copy=False)


def build_polytope_trajectory_cloud(
    phase_points: Array,
    *,
    anchor_index: int = 0,
    transition_rhos: tuple[float, ...] = (0.0, 1.0),
    margin: float = 0.02,
    event_radius: float = 1e-4,
) -> PolytopeTrajectoryCloud:
    """Build phase-wise irregular hulls with outward extension.

    Each non-event phase is a convex polytope cross-section around same-task
    demo points. The `margin` extends every hull outward, so all demonstration
    points sit inside with tolerance. Event phases collapse to the anchor point.
    """

    points = np.asarray(phase_points, dtype=np.float32)
    if points.ndim != 3:
        raise ValueError(f"phase_points must be [N, P, D], got {points.shape}.")
    if not 0 <= anchor_index < points.shape[0]:
        raise ValueError(f"anchor_index must be in [0, {points.shape[0]}), got {anchor_index}.")

    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise RuntimeError("scipy is required for polytope trajectory clouds.") from exc

    phase = normalized_phase(points.shape[1])
    transitions = tuple(float(np.clip(rho, 0.0, 1.0)) for rho in transition_rhos)
    event_indices = {int(np.argmin(np.abs(phase - rho))) for rho in transitions}
    sections: list[HullSection] = []
    for phase_index, rho in enumerate(phase):
        section_points = points[:, phase_index, :]
        equations = None
        is_event = phase_index in event_indices
        if not is_event:
            unique = np.unique(section_points, axis=0)
            if unique.shape[0] > section_points.shape[1]:
                try:
                    equations = ConvexHull(unique).equations.astype(np.float32, copy=False)
                except Exception:
                    equations = None
        sections.append(
            HullSection(
                phase=float(rho),
                points=section_points.astype(np.float32, copy=False),
                anchor=points[anchor_index, phase_index].astype(np.float32, copy=False),
                equations=equations,
                margin=float(margin),
                is_event=is_event,
            )
        )
    return PolytopeTrajectoryCloud(
        sections=tuple(sections),
        phase=phase,
        event_rhos=transitions,
        event_radius=event_radius,
        margin=margin,
    )


def polytope_section_distance(point: Array, section: HullSection, *, event_radius: float = 1e-4) -> float:
    """Distance outside one cross-section; zero means inside the cloud."""

    point = np.asarray(point, dtype=np.float32)
    if section.is_event:
        return float(np.linalg.norm(point - section.anchor) / max(event_radius, 1e-8))
    if section.equations is None:
        nearest = float(np.min(np.linalg.norm(section.points - point, axis=-1)))
        return max(0.0, nearest - section.margin) / max(section.margin, 1e-8)
    normals = section.equations[:, :-1]
    offsets = section.equations[:, -1]
    violations = normals @ point + offsets - section.margin
    return float(max(0.0, np.max(violations)) / max(section.margin, 1e-8))


def interpolated_polytope_distance(point: Array, cloud: PolytopeTrajectoryCloud, rho: float) -> float:
    """Interpolate cloud distance between neighboring discrete phase sections."""

    phase = cloud.phase
    rho = float(np.clip(rho, 0.0, 1.0))
    right = int(np.searchsorted(phase, rho, side="left"))
    if right <= 0:
        return polytope_section_distance(point, cloud.sections[0], event_radius=cloud.event_radius)
    if right >= len(phase):
        return polytope_section_distance(point, cloud.sections[-1], event_radius=cloud.event_radius)
    left = right - 1
    denom = max(float(phase[right] - phase[left]), 1e-8)
    weight = float((rho - phase[left]) / denom)
    left_distance = polytope_section_distance(point, cloud.sections[left], event_radius=cloud.event_radius)
    right_distance = polytope_section_distance(point, cloud.sections[right], event_radius=cloud.event_radius)
    return (1.0 - weight) * left_distance + weight * right_distance


def polytope_tube_loss(
    predicted: Array,
    cloud: PolytopeTrajectoryCloud,
    *,
    rho_start: float,
    config: TubeLossConfig = TubeLossConfig(),
) -> float:
    """Progress-window tube loss for irregular polytope trajectory clouds."""

    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.ndim != 2:
        raise ValueError(f"predicted must be [H, D], got {predicted.shape}.")
    losses = []
    for step, point in enumerate(predicted):
        phase_indices = progress_window_indices(cloud.phase, rho_start=rho_start, step=step, config=config)
        distance = min(
            polytope_section_distance(point, cloud.sections[int(index)], event_radius=cloud.event_radius)
            for index in phase_indices
        )
        losses.append(np.log1p(np.exp(distance / max(config.temperature, 1e-8))) ** 2)
    return float(np.mean(losses))


def olive_distance(point: Array, cloud: OliveTrajectoryCloud, phase_index: int) -> float:
    """Normalized distance to the anchor trajectory at one retained task-time phase.

    Other demonstrations determine cloud thickness, but do not contribute their
    own progress-indexed nearest points to the distance. This preserves the
    selected task-completion time structure while blurring cross-demo timing.
    """

    point = np.asarray(point, dtype=np.float32)
    radius = np.maximum(cloud.radius[phase_index], 1e-8)
    normalized = (point - cloud.anchor[phase_index]) / radius
    return float(np.linalg.norm(normalized) / np.sqrt(point.shape[-1]))


def progress_window_indices(
    phase: Array,
    *,
    rho_start: float,
    step: int,
    config: TubeLossConfig,
) -> Array:
    """Return phase indices in the allowed progress window for one predicted step."""

    low = rho_start + config.v_min * step * config.dt - config.delta
    high = rho_start + config.v_max * step * config.dt + config.delta
    mask = (phase >= max(0.0, low)) & (phase <= min(1.0, high))
    if np.any(mask):
        return np.flatnonzero(mask)
    nearest = int(np.argmin(np.abs(phase - np.clip((low + high) / 2.0, 0.0, 1.0))))
    return np.array([nearest], dtype=np.int64)


def interpolate_phase_trajectory(phase_points: Array, rho_values: Array) -> Array:
    """Sample a same-length weak anchor trajectory from phase-indexed points.

    This anchor is not the supervision target. It only provides a small in-cloud
    preference toward the original demonstration path after the strong cloud
    tube criterion has blurred fixed timestep matching.
    """

    phase_points = np.asarray(phase_points, dtype=np.float32)
    rho_values = np.asarray(rho_values, dtype=np.float32)
    if phase_points.ndim != 2:
        raise ValueError(f"phase_points must be [P, D], got {phase_points.shape}.")
    phase = normalized_phase(phase_points.shape[0])
    sampled = [
        np.interp(np.clip(rho_values, 0.0, 1.0), phase, phase_points[:, dim])
        for dim in range(phase_points.shape[1])
    ]
    return np.stack(sampled, axis=-1).astype(np.float32, copy=False)


def olive_tube_loss(
    predicted: Array,
    cloud: OliveTrajectoryCloud,
    *,
    rho_start: float,
    anchor: Array | None = None,
    config: TubeLossConfig = TubeLossConfig(),
) -> float:
    """Tube penalty with a small reward for staying near the original trajectory anchor."""

    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.ndim != 2:
        raise ValueError(f"predicted must be [H, D], got {predicted.shape}.")
    anchor_array = None if anchor is None else np.asarray(anchor, dtype=np.float32)
    if anchor_array is not None and anchor_array.shape != predicted.shape:
        raise ValueError(f"anchor must have shape {predicted.shape}, got {anchor_array.shape}.")

    losses = []
    for k, point in enumerate(predicted):
        phase_indices = progress_window_indices(cloud.phase, rho_start=rho_start, step=k, config=config)
        distance = min(olive_distance(point, cloud, int(phase_index)) for phase_index in phase_indices)
        penalty = np.log1p(np.exp((distance - 1.0) / max(config.temperature, 1e-8))) ** 2
        reward = 0.0
        if anchor_array is not None:
            anchor_dist = float(np.sum((point - anchor_array[k]) ** 2))
            reward = config.anchor_reward_weight * np.exp(
                -anchor_dist / max(config.anchor_sigma, 1e-8) ** 2
            )
        losses.append(penalty - reward)
    return float(np.mean(losses))


def event_channel_loss(
    predicted: Array,
    cloud: OliveTrajectoryCloud,
    *,
    rho_start: float,
    tube_config: TubeLossConfig = TubeLossConfig(),
    event_config: EventChannelConfig = EventChannelConfig(),
) -> float:
    """Strongly bind chunk points that cross anchor gripper-transition nodes.

    The olive tube allows exploration inside a phase. This loss carves narrow
    channels at the anchor transition points that connect adjacent phase clouds.
    It only applies to predicted steps whose allowed progress window touches an
    event phase such as 0 or 1.
    """

    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.ndim != 2:
        raise ValueError(f"predicted must be [H, D], got {predicted.shape}.")

    penalties = []
    for k, point in enumerate(predicted):
        phase_indices = progress_window_indices(cloud.phase, rho_start=rho_start, step=k, config=tube_config)
        for event_rho in event_config.event_rhos:
            event_index = int(np.argmin(np.abs(cloud.phase - np.clip(event_rho, 0.0, 1.0))))
            if event_index not in set(int(index) for index in phase_indices):
                continue
            event_anchor = cloud.anchor[event_index]
            normalized = (point - event_anchor) / max(event_config.radius, 1e-8)
            penalties.append(float(np.mean(normalized**2)))
    if not penalties:
        return 0.0
    return float(event_config.weight * np.mean(penalties))


def intra_chunk_smoothness_loss(actions: Array, config: SmoothLossConfig = SmoothLossConfig()) -> float:
    """Penalize velocity and acceleration jumps inside one action chunk."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be [H, D], got {actions.shape}.")
    if actions.shape[0] < 2:
        return 0.0
    velocity = np.diff(actions, axis=0)
    velocity_loss = float(np.mean(velocity**2))
    acceleration_loss = 0.0
    if velocity.shape[0] >= 2:
        acceleration = np.diff(velocity, axis=0)
        acceleration_loss = float(np.mean(acceleration**2))
    return config.velocity_weight * velocity_loss + config.acceleration_weight * acceleration_loss


def inter_chunk_smoothness_loss(
    previous: Array | None,
    current: Array,
    config: SmoothLossConfig = SmoothLossConfig(),
) -> float:
    """Penalize discontinuity between consecutive generated action chunks."""

    if previous is None:
        return 0.0
    previous = np.asarray(previous, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError(f"previous/current must be [H, D], got {previous.shape} and {current.shape}.")
    if previous.shape[-1] != current.shape[-1]:
        raise ValueError("previous and current chunks must share action dimension.")
    position_jump = current[0] - previous[-1]
    position_loss = float(np.mean(position_jump**2))
    velocity_loss = 0.0
    if previous.shape[0] >= 2 and current.shape[0] >= 2:
        previous_velocity = previous[-1] - previous[-2]
        current_velocity = current[1] - current[0]
        velocity_loss = float(np.mean((current_velocity - previous_velocity) ** 2))
    return (
        config.boundary_position_weight * position_loss
        + config.boundary_velocity_weight * velocity_loss
    )


def cloud_distance(point: Array, cloud_points: Array, sigma: float, eps: float = 1e-8) -> float:
    """Negative log density of a point under an isotropic Gaussian trajectory cloud."""

    point = np.asarray(point, dtype=np.float32)
    cloud_points = np.asarray(cloud_points, dtype=np.float32)
    squared = np.sum((cloud_points - point) ** 2, axis=-1)
    density = np.mean(np.exp(-squared / max(sigma, eps) ** 2))
    return float(-np.log(density + eps))


def tube_loss(
    predicted: Array,
    phase_cloud: Array,
    *,
    rho_start: float,
    config: TubeLossConfig = TubeLossConfig(),
) -> float:
    """Compute the progress-elastic tube loss for one predicted action chunk.

    `predicted` is [H, D]. `phase_cloud` is [N, P, D] for the current stage.
    """

    predicted = np.asarray(predicted, dtype=np.float32)
    phase_cloud = np.asarray(phase_cloud, dtype=np.float32)
    if predicted.ndim != 2:
        raise ValueError(f"predicted must be [H, D], got {predicted.shape}.")
    if phase_cloud.ndim != 3:
        raise ValueError(f"phase_cloud must be [N, P, D], got {phase_cloud.shape}.")
    if predicted.shape[-1] != phase_cloud.shape[-1]:
        raise ValueError("predicted and phase_cloud must share the same action dimension.")

    phase_grid = normalized_phase(phase_cloud.shape[1])
    penalties = []
    for k, point in enumerate(predicted):
        low = rho_start + config.v_min * k * config.dt - config.delta
        high = rho_start + config.v_max * k * config.dt + config.delta
        mask = (phase_grid >= max(0.0, low)) & (phase_grid <= min(1.0, high))
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(phase_grid - np.clip((low + high) / 2.0, 0.0, 1.0))))
            mask[nearest] = True
        candidates = phase_cloud[:, mask, :].reshape(-1, phase_cloud.shape[-1])
        distance = cloud_distance(point, candidates, config.sigma)
        margin = (distance - config.gamma) / max(config.temperature, 1e-8)
        penalties.append(np.log1p(np.exp(margin)) ** 2)
    return float(np.mean(penalties))


def tube_violation_rate(
    predicted: Array,
    phase_cloud: Array,
    *,
    rho_start: float,
    config: TubeLossConfig = TubeLossConfig(),
) -> float:
    """Fraction of predicted chunk points outside the allowed trajectory tube."""

    predicted = np.asarray(predicted, dtype=np.float32)
    phase_cloud = np.asarray(phase_cloud, dtype=np.float32)
    phase_grid = normalized_phase(phase_cloud.shape[1])
    violations = 0
    for k, point in enumerate(predicted):
        low = rho_start + config.v_min * k * config.dt - config.delta
        high = rho_start + config.v_max * k * config.dt + config.delta
        mask = (phase_grid >= max(0.0, low)) & (phase_grid <= min(1.0, high))
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(phase_grid - np.clip((low + high) / 2.0, 0.0, 1.0))))
            mask[nearest] = True
        candidates = phase_cloud[:, mask, :].reshape(-1, phase_cloud.shape[-1])
        distance = cloud_distance(point, candidates, config.sigma)
        violations += int(distance > config.gamma)
    return violations / max(len(predicted), 1)


def progress_backward_rate(predicted_rho: Array) -> float:
    """Fraction of adjacent progress predictions that move backward."""

    predicted_rho = np.asarray(predicted_rho, dtype=np.float32)
    if predicted_rho.ndim != 1:
        raise ValueError(f"predicted_rho must be [H], got {predicted_rho.shape}.")
    if len(predicted_rho) < 2:
        return 0.0
    return float(np.mean(np.diff(predicted_rho) < 0.0))
