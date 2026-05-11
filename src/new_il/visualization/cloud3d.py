from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from new_il.patcs import build_olive_trajectory_cloud


STAGE_COLORS = ["#5b8fd9", "#70ad47", "#c55a11", "#8064a2", "#4bacc6", "#c0504d"]


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "phase_points" in data:
            return data["phase_points"]
        if "actions" in data:
            return data["actions"]
        first_key = sorted(data.files)[0]
        return data[first_key]
    raise SystemExit("Use a .npy or .npz file containing [N, P, D] phase points.")


def _axis_equal_3d(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _expanded_hull_points(points: np.ndarray, margin: float, contraction: float, anchor: np.ndarray) -> np.ndarray:
    center = anchor.reshape(1, 3)
    offsets = points - center
    norms = np.linalg.norm(offsets, axis=-1, keepdims=True)
    scale = contraction * (1.0 + margin / np.maximum(norms, 1e-6))
    return center + offsets * scale


def _add_irregular_phase_hull(
    ax,
    points: np.ndarray,
    *,
    color: str,
    alpha: float,
    margin: float = 0.0,
    contraction: float = 1.0,
    anchor: np.ndarray | None = None,
) -> None:
    if margin > 0:
        if anchor is None:
            anchor = points.mean(axis=0)
        points = _expanded_hull_points(points, margin, contraction, anchor)
    if points.shape[0] < 4:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color=color, alpha=alpha)
        return
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color=color, alpha=alpha)
        return
    try:
        hull = ConvexHull(points)
    except Exception:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color=color, alpha=alpha)
        return
    ax.plot_trisurf(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        triangles=hull.simplices,
        color=color,
        alpha=alpha,
        linewidth=0.2,
        edgecolor=color,
        shade=False,
    )


def _plot_colored_line(ax, xyz: np.ndarray, values: np.ndarray, *, alpha: float, linewidth: float) -> None:
    cmap = colormaps["viridis"]
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    collection = Line3DCollection(
        segments,
        colors=cmap(values[:-1]),
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(collection)


def render_cloud(
    points: np.ndarray,
    output: Path,
    dims: tuple[int, int, int],
    *,
    gripper_states: np.ndarray | None = None,
    continuous: bool = True,
    hull_margin: float = 0.012,
    view: tuple[float, float] | None = None,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3:
        stages = points[None, ...]
    elif points.ndim == 4:
        stages = points
    else:
        raise SystemExit(f"Expected [N, P, D] or [R, N, P, D], got {points.shape}.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install visualization deps: uv sync --extra viz") from exc

    x_dim, y_dim, z_dim = dims
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_xyz = []
    cumulative_offsets = np.concatenate([[0], np.cumsum([stage.shape[1] for stage in stages[:-1]])])
    total_phase_steps = int(sum(stage.shape[1] for stage in stages))
    gripper_array = None if gripper_states is None else np.asarray(gripper_states, dtype=np.float32)
    for stage_index, stage_points in enumerate(stages):
        stage_base = int(cumulative_offsets[stage_index])
        color = STAGE_COLORS[stage_index % len(STAGE_COLORS)]
        cloud = build_olive_trajectory_cloud(stage_points)
        xyz_points = stage_points[:, :, [x_dim, y_dim, z_dim]]
        all_xyz.append(xyz_points.reshape(-1, 3))
        stage_progress = (stage_base + np.arange(stage_points.shape[1])) / max(total_phase_steps - 1, 1)
        for demo_index, demo in enumerate(stage_points):
            xyz = demo[:, [x_dim, y_dim, z_dim]]
            if continuous:
                _plot_colored_line(ax, xyz, stage_progress, alpha=0.32, linewidth=0.9)
            else:
                ax.plot(
                    demo[:, x_dim],
                    demo[:, y_dim],
                    demo[:, z_dim],
                    color=color,
                    alpha=0.28,
                    linewidth=0.9,
                    label=f"stage {stage_index} demos" if demo_index == 0 else None,
                )
        anchor_xyz = cloud.anchor[:, [x_dim, y_dim, z_dim]]
        if continuous:
            _plot_colored_line(ax, anchor_xyz, stage_progress, alpha=0.95, linewidth=2.8)
        else:
            ax.plot(
                cloud.anchor[:, x_dim],
                cloud.anchor[:, y_dim],
                cloud.anchor[:, z_dim],
                color="#1d1d1f",
                linewidth=2.2,
                label=f"stage {stage_index} anchor",
            )
        phase_stride = max(1, stage_points.shape[1] // 12)
        for phase_index in range(0, stage_points.shape[1], phase_stride):
            phase_color = colormaps["viridis"](stage_progress[phase_index])
            raw_points = xyz_points[:, phase_index, :]
            contraction = float(cloud.contraction[phase_index, 0])
            anchor = anchor_xyz[phase_index]
            _add_irregular_phase_hull(
                ax,
                raw_points,
                color=phase_color,
                alpha=0.06,
                margin=0.0,
            )
            _add_irregular_phase_hull(
                ax,
                raw_points,
                color=phase_color,
                alpha=0.16,
                margin=hull_margin,
                contraction=contraction,
                anchor=anchor,
            )
        if gripper_array is not None:
            start_state = int(gripper_array[stage_index, 0] > 0.5)
            end_state = int(gripper_array[stage_index, 1] > 0.5)
            start_color = "#d62728" if start_state == 0 else "#1f77b4"
            end_color = "#d62728" if end_state == 0 else "#1f77b4"
            ax.scatter(
                stage_points[:, 0, x_dim],
                stage_points[:, 0, y_dim],
                stage_points[:, 0, z_dim],
                color=start_color,
                s=26,
                alpha=0.9,
                label="gripper 0/open" if stage_index == 0 and start_state == 0 else None,
            )
            ax.scatter(
                stage_points[:, -1, x_dim],
                stage_points[:, -1, y_dim],
                stage_points[:, -1, z_dim],
                color=end_color,
                s=26,
                alpha=0.9,
                label="gripper 1/closed" if stage_index == 0 and end_state == 1 else None,
            )

    ax.set_xlabel(f"dim {x_dim}")
    ax.set_ylabel(f"dim {y_dim}")
    ax.set_zlabel(f"dim {z_dim}")
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    _axis_equal_3d(ax, np.concatenate(all_xyz, axis=0))
    if continuous:
        mappable = plt.cm.ScalarMappable(cmap="viridis")
        mappable.set_array([0.0, 1.0])
        fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="task progress")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    summary = {
        "points_shape": list(points.shape),
        "stage_count": int(stages.shape[0]),
        "dims": list(dims),
        "output": str(output),
        "cloud_surface": "irregular per-phase convex hulls from same-task demo points",
        "visualized_hull_margin": hull_margin,
        "continuous_progress_coloring": continuous,
        "gripper_state_colors": "red=0/open, blue=1/closed" if gripper_states is not None else None,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 3D olive trajectory cloud from phase points.")
    parser.add_argument("--phase-points", type=Path, required=True, help=".npy/.npz shaped [N, P, D].")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dims", nargs=3, type=int, default=(0, 1, 2))
    parser.add_argument("--gripper-states", type=Path)
    parser.add_argument("--hull-margin", type=float, default=0.012)
    parser.add_argument("--view", nargs=2, type=float)
    parser.add_argument("--stage-colors", action="store_true")
    args = parser.parse_args()
    gripper_states = None if args.gripper_states is None else np.load(args.gripper_states)
    render_cloud(
        _load_array(args.phase_points),
        args.output,
        tuple(args.dims),
        gripper_states=gripper_states,
        continuous=not args.stage_colors,
        hull_margin=args.hull_margin,
        view=None if args.view is None else (args.view[0], args.view[1]),
    )
