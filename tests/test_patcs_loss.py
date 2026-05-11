from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from new_il.training.patcs_loss import (
    PatcsArtifact,
    load_patcs_artifact,
    patcs_chunk_loss,
    patcs_event_loss,
    patcs_total_loss,
)
from new_il.patcs import TubeLossConfig


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_artifact(
    num_stages: int = 2,
    num_demos: int = 5,
    num_phase: int = 8,
    state_dim: int = 3,
    rng: np.random.Generator | None = None,
) -> PatcsArtifact:
    """Build a minimal in-memory PatcsArtifact for testing (no file I/O)."""

    if rng is None:
        rng = np.random.default_rng(42)

    phase_points = rng.standard_normal((num_stages, num_demos, num_phase, state_dim)).astype(np.float32)
    anchor = phase_points[:, 0, :, :]  # first demo is anchor, shape [S, P, D]

    # Build simple hull equations: 2*(D+1) halfspace rows per phase (box around points).
    max_eq = 2 * state_dim
    hull_equations = np.zeros((num_stages, num_phase, max_eq, state_dim + 1), dtype=np.float32)
    hull_equation_counts = np.zeros((num_stages, num_phase), dtype=np.int32)

    for s in range(num_stages):
        for p in range(num_phase):
            pts = phase_points[s, :, p, :]  # [N, D]
            lo = pts.min(axis=0) - 0.5
            hi = pts.max(axis=0) + 0.5
            rows = []
            for d in range(state_dim):
                n_pos = np.zeros(state_dim, dtype=np.float32)
                n_pos[d] = 1.0
                rows.append(np.append(n_pos, -hi[d]))   # n·x - hi <= 0
                n_neg = np.zeros(state_dim, dtype=np.float32)
                n_neg[d] = -1.0
                rows.append(np.append(n_neg, lo[d]))    # -n·x + lo <= 0
            eqs = np.stack(rows, axis=0).astype(np.float32)
            count = eqs.shape[0]
            hull_equations[s, p, :count] = eqs
            hull_equation_counts[s, p] = count

    event_mask = np.zeros((num_stages, num_phase), dtype=bool)
    event_mask[:, 0] = True   # first phase is event
    event_mask[:, -1] = True  # last phase is event

    phase_grid = np.linspace(0.0, 1.0, num_phase, dtype=np.float32)

    return PatcsArtifact(
        phase_points=phase_points,
        anchor=anchor,
        hull_equations=hull_equations,
        hull_equation_counts=hull_equation_counts,
        event_mask=event_mask,
        phase_grid=phase_grid,
        margin=0.5,
        event_radius=0.1,  # large enough for test assertions
        num_stages=num_stages,
        num_phase=num_phase,
        state_dim=state_dim,
    )


# ---------------------------------------------------------------------------
# Load from real artifact file (skipped if not present)
# ---------------------------------------------------------------------------

ARTIFACT_PATH = Path("data/patcs_artifacts/libero_object/orange_juice_basket_ee_pos_patcs.npz")


@pytest.mark.skipif(not ARTIFACT_PATH.exists(), reason="real artifact not present")
def test_load_real_artifact() -> None:
    artifact = load_patcs_artifact(ARTIFACT_PATH)
    assert artifact.num_stages == 3
    assert artifact.num_phase == 64
    assert artifact.state_dim == 3
    assert artifact.phase_grid.shape == (64,)
    assert artifact.anchor.shape == (3, 64, 3)
    assert artifact.hull_equations.shape[0] == 3
    assert artifact.event_mask.shape == (3, 64)
    assert artifact.event_mask[:, 0].all()
    assert artifact.event_mask[:, -1].all()


def test_load_missing_artifact_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_patcs_artifact(Path("does/not/exist.npz"))


def test_load_bad_artifact_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    np.savez(bad, phase_points=np.zeros((2, 4, 8, 3)))
    with pytest.raises(KeyError):
        load_patcs_artifact(bad)


# ---------------------------------------------------------------------------
# Round-trip via tmp npz
# ---------------------------------------------------------------------------

def _save_artifact(artifact: PatcsArtifact, path: Path) -> None:
    np.savez_compressed(
        path,
        phase_points=artifact.phase_points,
        anchor=artifact.anchor,
        hull_equations=artifact.hull_equations,
        hull_equation_counts=artifact.hull_equation_counts,
        event_mask=artifact.event_mask,
        phase_grid=artifact.phase_grid,
        margin=np.array(artifact.margin, dtype=np.float32),
        event_radius=np.array(artifact.event_radius, dtype=np.float32),
    )


def test_load_roundtrip(tmp_path: Path) -> None:
    original = _make_artifact()
    npz = tmp_path / "test.npz"
    _save_artifact(original, npz)
    loaded = load_patcs_artifact(npz)
    assert loaded.num_stages == original.num_stages
    assert loaded.num_phase == original.num_phase
    assert loaded.state_dim == original.state_dim
    np.testing.assert_allclose(loaded.phase_grid, original.phase_grid)
    np.testing.assert_allclose(loaded.anchor, original.anchor)


# ---------------------------------------------------------------------------
# patcs_chunk_loss
# ---------------------------------------------------------------------------

def test_chunk_loss_is_finite() -> None:
    artifact = _make_artifact()
    rng = np.random.default_rng(0)
    predicted = rng.standard_normal((4, 3)).astype(np.float32)
    loss = patcs_chunk_loss(predicted, artifact, stage=0, rho_start=0.0)
    assert np.isfinite(loss)


def test_chunk_loss_anchor_lower_than_random() -> None:
    """Anchor trajectory should incur lower tube loss than random predictions."""

    artifact = _make_artifact(num_phase=16)
    config = TubeLossConfig(v_min=0.0, v_max=0.5, delta=0.1, dt=1.0 / 16)
    rng = np.random.default_rng(7)

    # Build a short chunk from the anchor itself (sampled at rho ≈ 0)
    anchor_chunk = artifact.anchor[0, :4, :]  # [4, D] first 4 anchor points
    random_chunk = rng.standard_normal((4, artifact.state_dim)).astype(np.float32) * 5.0

    anchor_loss = patcs_chunk_loss(anchor_chunk, artifact, stage=0, rho_start=0.0, config=config)
    random_loss = patcs_chunk_loss(random_chunk, artifact, stage=0, rho_start=0.0, config=config)
    assert anchor_loss < random_loss


def test_chunk_loss_bad_stage_raises() -> None:
    artifact = _make_artifact()
    predicted = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="stage"):
        patcs_chunk_loss(predicted, artifact, stage=99, rho_start=0.0)


def test_chunk_loss_bad_dim_raises() -> None:
    artifact = _make_artifact(state_dim=3)
    predicted = np.zeros((4, 5), dtype=np.float32)  # wrong dim
    with pytest.raises(ValueError, match="dim"):
        patcs_chunk_loss(predicted, artifact, stage=0, rho_start=0.0)


def test_chunk_loss_bad_shape_raises() -> None:
    artifact = _make_artifact()
    with pytest.raises(ValueError, match="H, D"):
        patcs_chunk_loss(np.zeros((3,), dtype=np.float32), artifact, stage=0, rho_start=0.0)


# ---------------------------------------------------------------------------
# patcs_event_loss
# ---------------------------------------------------------------------------

def test_event_loss_anchor_at_event_is_small() -> None:
    """Predicting the exact anchor point at an event phase yields near-zero event loss."""

    artifact = _make_artifact(num_phase=8, state_dim=3)
    # Use a tight config so only phase 0 (rho=0) is in the progress window,
    # not phase 7 (rho=1) whose anchor point differs from phase 0's anchor.
    tight_config = TubeLossConfig(v_min=0.0, v_max=0.05, delta=0.01, dt=1.0 / 8)
    anchor_pt = artifact.anchor[0, 0, :]  # [D]  exact anchor at phase 0
    predicted = np.stack([anchor_pt] * 4, axis=0).astype(np.float32)  # [4, D]
    loss = patcs_event_loss(predicted, artifact, stage=0, rho_start=0.0, config=tight_config)
    assert loss < 1.0


def test_event_loss_far_point_large() -> None:
    """A point far from the anchor at an event phase incurs large event loss."""

    artifact = _make_artifact(num_phase=8, state_dim=3)
    anchor_pt = artifact.anchor[0, 0, :]
    far = anchor_pt + 100.0  # very far
    predicted = np.stack([far] * 4, axis=0).astype(np.float32)
    loss = patcs_event_loss(predicted, artifact, stage=0, rho_start=0.0, event_weight=10.0)
    assert loss > 1.0


def test_event_loss_no_event_phases_returns_zero() -> None:
    artifact = _make_artifact(num_phase=8)
    # Override event_mask to all-False.
    no_event = PatcsArtifact(
        phase_points=artifact.phase_points,
        anchor=artifact.anchor,
        hull_equations=artifact.hull_equations,
        hull_equation_counts=artifact.hull_equation_counts,
        event_mask=np.zeros_like(artifact.event_mask),
        phase_grid=artifact.phase_grid,
        margin=artifact.margin,
        event_radius=artifact.event_radius,
        num_stages=artifact.num_stages,
        num_phase=artifact.num_phase,
        state_dim=artifact.state_dim,
    )
    predicted = np.zeros((4, artifact.state_dim), dtype=np.float32)
    loss = patcs_event_loss(predicted, no_event, stage=0, rho_start=0.0)
    assert loss == 0.0


# ---------------------------------------------------------------------------
# patcs_total_loss
# ---------------------------------------------------------------------------

def test_total_loss_keys() -> None:
    artifact = _make_artifact()
    rng = np.random.default_rng(1)
    predicted = rng.standard_normal((4, 3)).astype(np.float32) * 0.1
    result = patcs_total_loss(predicted, artifact, stage=0, rho_start=0.5)
    assert set(result.keys()) == {"tube", "event", "total"}
    assert np.isfinite(result["total"])
    assert abs(result["total"] - (result["tube"] + result["event"])) < 1e-5


def test_total_loss_all_finite() -> None:
    artifact = _make_artifact()
    rng = np.random.default_rng(3)
    predicted = rng.standard_normal((8, 3)).astype(np.float32)
    result = patcs_total_loss(predicted, artifact, stage=1, rho_start=0.3)
    for key, val in result.items():
        assert np.isfinite(val), f"{key} is not finite"


@pytest.mark.skipif(not ARTIFACT_PATH.exists(), reason="real artifact not present")
def test_total_loss_real_artifact() -> None:
    artifact = load_patcs_artifact(ARTIFACT_PATH)
    rng = np.random.default_rng(99)
    predicted = rng.standard_normal((8, artifact.state_dim)).astype(np.float32) * 0.1
    for stage in range(artifact.num_stages):
        result = patcs_total_loss(predicted, artifact, stage=stage, rho_start=0.0)
        assert result["total"] >= 0.0
        for val in result.values():
            assert np.isfinite(val), f"stage {stage}: non-finite loss"
