from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from new_il.patcs import TubeLossConfig, progress_window_indices


@dataclass(frozen=True)
class PatcsArtifact:
    """Arrays loaded from a precomputed PA-TCS .npz artifact.

    All arrays are numpy float32 / int32 / bool. The trainer loads this once
    and queries it per batch item without re-running scipy.
    """

    phase_points: np.ndarray       # [S, N, P, D]
    anchor: np.ndarray             # [S, P, D]
    hull_equations: np.ndarray     # [S, P, E, D+1]  halfspace normals + offset
    hull_equation_counts: np.ndarray  # [S, P]  number of valid rows in E
    event_mask: np.ndarray         # [S, P]  bool; True = anchor-only constraint
    phase_grid: np.ndarray         # [P]
    margin: float
    event_radius: float
    num_stages: int
    num_phase: int
    state_dim: int


def load_patcs_artifact(path: Path | str) -> PatcsArtifact:
    """Load and validate a PA-TCS .npz artifact produced by build_patcs_artifact."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PATCS artifact not found: {path}")
    data = np.load(path, allow_pickle=False)

    required = {
        "phase_points", "anchor", "hull_equations", "hull_equation_counts",
        "event_mask", "phase_grid",
    }
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Artifact {path.name} missing arrays: {sorted(missing)}")

    phase_points = data["phase_points"].astype(np.float32, copy=False)  # [S, N, P, D]
    anchor = data["anchor"].astype(np.float32, copy=False)              # [S, P, D]
    hull_equations = data["hull_equations"].astype(np.float32, copy=False)
    hull_equation_counts = data["hull_equation_counts"].astype(np.int32, copy=False)
    event_mask = data["event_mask"].astype(bool, copy=False)
    phase_grid = data["phase_grid"].astype(np.float32, copy=False)

    if phase_points.ndim != 4:
        raise ValueError(f"phase_points must be [S, N, P, D], got {phase_points.shape}")
    s, _, p, d = phase_points.shape
    if anchor.shape != (s, p, d):
        raise ValueError(f"anchor shape {anchor.shape} inconsistent with phase_points {phase_points.shape}")
    if hull_equations.shape[:2] != (s, p):
        raise ValueError(f"hull_equations shape {hull_equations.shape} inconsistent with [S={s}, P={p}]")
    if hull_equation_counts.shape != (s, p):
        raise ValueError(f"hull_equation_counts shape {hull_equation_counts.shape} != ({s}, {p})")
    if event_mask.shape != (s, p):
        raise ValueError(f"event_mask shape {event_mask.shape} != ({s}, {p})")
    if phase_grid.shape != (p,):
        raise ValueError(f"phase_grid shape {phase_grid.shape} != ({p},)")

    margin = float(data["margin"]) if "margin" in data.files else float(data.get("hull_margin", 0.012))
    event_radius = float(data["event_radius"]) if "event_radius" in data.files else 1e-4

    return PatcsArtifact(
        phase_points=phase_points,
        anchor=anchor,
        hull_equations=hull_equations,
        hull_equation_counts=hull_equation_counts,
        event_mask=event_mask,
        phase_grid=phase_grid,
        margin=margin,
        event_radius=event_radius,
        num_stages=s,
        num_phase=p,
        state_dim=d,
    )


def _hull_signed_distance(
    point: np.ndarray,
    equations: np.ndarray,
    count: int,
    margin: float,
) -> float:
    """Largest halfspace violation minus margin; negative means inside."""

    if count <= 0:
        return 0.0
    eqs = equations[:count]
    violations = eqs[:, :-1] @ point + eqs[:, -1]
    return float(np.max(violations) - margin)


def _section_distance(
    point: np.ndarray,
    artifact: PatcsArtifact,
    stage: int,
    phase_idx: int,
) -> float:
    """Distance outside one phase cross-section; 0 means inside.

    Event phases return Euclidean distance to the anchor point normalized by
    event_radius. Non-event phases use precomputed hull halfspace equations.
    """

    if artifact.event_mask[stage, phase_idx]:
        dist = float(np.linalg.norm(point - artifact.anchor[stage, phase_idx]))
        return dist / max(artifact.event_radius, 1e-8)

    count = int(artifact.hull_equation_counts[stage, phase_idx])
    if count == 0:
        # No valid hull — fall back to nearest demo point distance.
        demo_points = artifact.phase_points[stage, :, phase_idx, :]
        nearest = float(np.min(np.linalg.norm(demo_points - point, axis=-1)))
        return max(0.0, nearest - artifact.margin) / max(artifact.margin, 1e-8)

    raw = _hull_signed_distance(
        point, artifact.hull_equations[stage, phase_idx], count, artifact.margin
    )
    return float(max(0.0, raw)) / max(artifact.margin, 1e-8)


def patcs_chunk_loss(
    predicted: np.ndarray,
    artifact: PatcsArtifact,
    stage: int,
    rho_start: float,
    config: TubeLossConfig = TubeLossConfig(),
) -> float:
    """Online PA-TCS tube loss for one predicted action chunk.

    Uses precomputed hull equations from the artifact; no scipy at query time.

    Args:
        predicted: [H, D] predicted action chunk (continuous dims only).
        artifact: loaded PatcsArtifact for this task.
        stage: stage index (0-indexed, selects which phase cloud to query).
        rho_start: progress phase at the start of this chunk in [0, 1].
        config: tube loss numerical config.

    Returns:
        Scalar loss value.
    """

    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.ndim != 2:
        raise ValueError(f"predicted must be [H, D], got {predicted.shape}")
    if not 0 <= stage < artifact.num_stages:
        raise ValueError(f"stage {stage} out of range [0, {artifact.num_stages})")
    if predicted.shape[-1] != artifact.state_dim:
        raise ValueError(
            f"predicted dim {predicted.shape[-1]} != artifact state_dim {artifact.state_dim}"
        )

    penalties = []
    for k, point in enumerate(predicted):
        phase_indices = progress_window_indices(
            artifact.phase_grid, rho_start=rho_start, step=k, config=config
        )
        distance = min(
            _section_distance(point, artifact, stage, int(idx)) for idx in phase_indices
        )
        margin_val = (distance - 1.0) / max(config.temperature, 1e-8)
        # numerically stable softplus: logaddexp(0, x) = log(1 + exp(x))
        penalties.append(float(np.logaddexp(0.0, margin_val) ** 2))

    return float(np.mean(penalties))


def patcs_event_loss(
    predicted: np.ndarray,
    artifact: PatcsArtifact,
    stage: int,
    rho_start: float,
    config: TubeLossConfig = TubeLossConfig(),
    event_weight: float = 10.0,
) -> float:
    """Strong anchor constraint for predicted steps whose window touches event phases.

    Only fires when the progress window for a predicted step includes a phase
    marked as an event (event_mask == True). Applies MSE to the anchor point
    normalized by event_radius.

    Args:
        predicted: [H, D] predicted action chunk.
        artifact: loaded PatcsArtifact.
        stage: stage index.
        rho_start: progress phase at chunk start.
        config: tube loss config (for progress window calculation).
        event_weight: multiplier on the event penalty.

    Returns:
        Scalar event loss (0 if no event phases are in the progress window).
    """

    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.ndim != 2:
        raise ValueError(f"predicted must be [H, D], got {predicted.shape}")
    if not 0 <= stage < artifact.num_stages:
        raise ValueError(f"stage {stage} out of range [0, {artifact.num_stages})")

    event_indices = set(int(i) for i in np.flatnonzero(artifact.event_mask[stage]))
    if not event_indices:
        return 0.0

    penalties = []
    for k, point in enumerate(predicted):
        phase_indices = progress_window_indices(
            artifact.phase_grid, rho_start=rho_start, step=k, config=config
        )
        for phase_idx in phase_indices:
            if int(phase_idx) not in event_indices:
                continue
            anchor = artifact.anchor[stage, int(phase_idx)]
            normalized = (point - anchor) / max(artifact.event_radius, 1e-8)
            penalties.append(float(np.mean(normalized ** 2)))

    if not penalties:
        return 0.0
    return float(event_weight * np.mean(penalties))


def patcs_total_loss(
    predicted: np.ndarray,
    artifact: PatcsArtifact,
    stage: int,
    rho_start: float,
    config: TubeLossConfig = TubeLossConfig(),
    event_weight: float = 10.0,
    tube_weight: float = 1.0,
) -> dict[str, float]:
    """Combined PA-TCS loss: tube + event channel.

    Returns a dict with keys ``tube``, ``event``, and ``total`` so the trainer
    can log each component separately.
    """

    tube = tube_weight * patcs_chunk_loss(predicted, artifact, stage, rho_start, config)
    event = patcs_event_loss(predicted, artifact, stage, rho_start, config, event_weight)
    return {"tube": tube, "event": event, "total": tube + event}
