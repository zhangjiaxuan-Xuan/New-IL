from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from new_il.patcs import gripper_transition_indices
from new_il.patcs_probabilistic import (
    ProbabilisticTubeConfig,
    ProbabilisticTrajectoryTube,
    build_probabilistic_trajectory_tube,
    tube_surface_mesh,
    tube_radius_diagnostics,
)


@dataclass(frozen=True)
class RolloutRecord:
    path: Path
    language: str
    state: np.ndarray
    actions: np.ndarray
    transitions: np.ndarray


def _load_records_from_manifest(
    manifest: Path,
    *,
    suite: str,
    task: str,
    max_demos: int,
) -> list[RolloutRecord]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    records = []
    for item in data["records"]:
        if not item.get("success", False):
            continue
        if item.get("task_suite_name") != suite or f"task_{int(item.get('task_idx')):02d}" != task:
            continue
        path = Path(item["source_path"])
        loaded = _load_rollout(path)
        records.append(loaded)
        if len(records) >= max_demos:
            break
    if not records:
        raise SystemExit(f"No successful records found for {suite}/{task} in {manifest}")
    return records


def _load_rollout(path: Path) -> RolloutRecord:
    data = np.load(path, allow_pickle=False)
    if "observation.state" not in data or "actions" not in data:
        raise SystemExit(f"{path} missing observation.state or actions")
    state = np.asarray(data["observation.state"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if state.shape[0] != actions.shape[0]:
        raise SystemExit(f"{path} state/actions length mismatch: {state.shape[0]} vs {actions.shape[0]}")
    language = str(data["language"].item()) if "language" in data and data["language"].shape == () else path.stem
    transitions = np.asarray(
        gripper_transition_indices(actions, gripper_index=-1, threshold=0.0),
        dtype=np.int32,
    )
    return RolloutRecord(path=path, language=language, state=state, actions=actions, transitions=transitions)


def _resample_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must be [T, D], got {values.shape}")
    if len(values) == target_len:
        return values.copy()
    src_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.stack([np.interp(dst_x, src_x, values[:, d]) for d in range(values.shape[1])], axis=-1).astype(np.float32)


def _condition_to_target_events(
    base_cloud: np.ndarray,
    target_xyz: np.ndarray,
    target_transitions: np.ndarray,
    *,
    event_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    conditioned = base_cloud.copy()
    target_len = target_xyz.shape[0]
    event_mask = np.zeros((target_len,), dtype=bool)
    for transition in target_transitions:
        center = int(np.clip(transition, 0, target_len - 1))
        lo = max(0, center - event_window)
        hi = min(target_len, center + event_window + 1)
        if hi <= lo:
            continue
        span = max(center - lo, hi - center - 1, 1)
        for t in range(lo, hi):
            dist = abs(t - center)
            # At the transition, all demos collapse exactly to target observation.
            # Nearby points blend back into the full cloud for visual inspection.
            weight = 1.0 - min(dist / span, 1.0)
            conditioned[:, t, :] = (1.0 - weight) * conditioned[:, t, :] + weight * target_xyz[t]
            event_mask[t] = True
    return conditioned.astype(np.float32, copy=False), event_mask


def build_demo_conditioned_cloud(
    records: list[RolloutRecord],
    *,
    target_index: int,
    dims: tuple[int, int, int],
    event_window: int,
    tube_config: ProbabilisticTubeConfig | None = None,
) -> dict[str, np.ndarray | list[str] | int | str | ProbabilisticTrajectoryTube]:
    if not 0 <= target_index < len(records):
        raise SystemExit(f"target_index must be in [0, {len(records)}), got {target_index}")
    target = records[target_index]
    target_xyz = target.state[:, list(dims)]
    base_cloud = np.stack([
        _resample_to_length(record.state[:, list(dims)], len(target_xyz)) for record in records
    ], axis=0)
    source_gripper = np.stack([
        _resample_to_length(record.actions[:, [-1]], len(target_xyz))[:, 0] for record in records
    ], axis=0)
    conditioned_cloud, event_mask = _condition_to_target_events(
        base_cloud,
        target_xyz,
        target.transitions,
        event_window=event_window,
    )
    tube_cfg = tube_config or ProbabilisticTubeConfig(event_window=event_window)
    tube = build_probabilistic_trajectory_tube(
        [record.state[:, list(dims)] for record in records],
        target_index=target_index,
        transitions=target.transitions,
        config=tube_cfg,
    )
    return {
        "base_cloud": base_cloud,
        "conditioned_cloud": conditioned_cloud,
        "tube": tube,
        "target_xyz": target_xyz.astype(np.float32, copy=False),
        "event_mask": event_mask,
        "target_transitions": target.transitions.astype(np.int32, copy=False),
        "target_gripper": target.actions[:, -1].astype(np.float32, copy=False),
        "source_gripper": source_gripper.astype(np.float32, copy=False),
        "source_lengths": np.asarray([len(record.state) for record in records], dtype=np.int32),
        "source_transition_counts": np.asarray([len(record.transitions) for record in records], dtype=np.int32),
        "source_paths": [str(record.path) for record in records],
        "target_path": str(target.path),
        "target_language": target.language,
    }


def _plot_colored_line(ax, xyz: np.ndarray, values: np.ndarray, *, alpha: float, linewidth: float, cmap_name: str) -> None:
    cmap = colormaps[cmap_name]
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    collection = Line3DCollection(segments, colors=cmap(values[:-1]), linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(collection)


def _add_tube_surface(ax, tube: ProbabilisticTrajectoryTube, *, stride: int = 3) -> None:
    vertices, faces, density = tube_surface_mesh(tube, stride=stride)
    if len(faces) == 0:
        return
    face_sections = np.arange(len(faces), dtype=np.int32) // (2 * tube.config.surface_sides)
    section_density = density[np.clip(face_sections, 0, len(density) - 1)]
    density_norm = (section_density - section_density.min()) / max(float(np.ptp(section_density)), 1e-8)
    colors = colormaps["viridis"](density_norm)
    colors[:, 3] = 0.18 + 0.22 * density_norm
    mesh = Poly3DCollection(vertices[faces], facecolors=colors, edgecolors="none", linewidths=0.0)
    ax.add_collection3d(mesh)


def _ellipse_points_from_cov(mean2: np.ndarray, cov2: np.ndarray, *, radius: float, count: int = 72) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 1e-12)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=True, dtype=np.float32)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return mean2 + (unit @ vecs.T) * (np.sqrt(vals)[None, :] * radius)


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


def _axis_equal_2d(ax, xy: np.ndarray, *, pad: float = 0.05) -> None:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    radius *= 1.0 + pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def render_demo_conditioned_cloud(
    cloud: dict[str, np.ndarray | list[str] | int | str],
    output: Path,
    *,
    view: tuple[float, float] | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install visualization deps: uv sync --extra viz") from exc

    base_cloud = np.asarray(cloud["base_cloud"], dtype=np.float32)
    conditioned_cloud = np.asarray(cloud["conditioned_cloud"], dtype=np.float32)
    target_xyz = np.asarray(cloud["target_xyz"], dtype=np.float32)
    transitions = np.asarray(cloud["target_transitions"], dtype=np.int32)
    progress = np.linspace(0.0, 1.0, target_xyz.shape[0], dtype=np.float32)
    diag = tube_radius_diagnostics(tube)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for demo in base_cloud:
        _plot_colored_line(ax, demo, progress, alpha=0.12, linewidth=0.8, cmap_name="Greys")
    for demo in conditioned_cloud:
        _plot_colored_line(ax, demo, progress, alpha=0.26, linewidth=0.9, cmap_name="viridis")
    _plot_colored_line(ax, target_xyz, progress, alpha=1.0, linewidth=3.0, cmap_name="plasma")

    if len(transitions):
        event_xyz = target_xyz[np.clip(transitions, 0, len(target_xyz) - 1)]
        ax.scatter(event_xyz[:, 0], event_xyz[:, 1], event_xyz[:, 2], color="#d62728", s=48, label="target gripper transitions")

    ax.set_xlabel("ee x")
    ax.set_ylabel("ee y")
    ax.set_zlabel("ee z")
    ax.set_title("Demo-conditioned full-trajectory cloud")
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    _axis_equal_3d(ax, np.concatenate([base_cloud.reshape(-1, 3), conditioned_cloud.reshape(-1, 3)], axis=0))
    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array([0.0, 1.0])
    fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08, label="target demo progress")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _render_cloud_row(
    fig,
    grid,
    row: int,
    cloud: dict[str, np.ndarray | list[str] | int | str],
    *,
    title_prefix: str,
    view: tuple[float, float] | None,
) -> None:
    base_cloud = np.asarray(cloud["base_cloud"], dtype=np.float32)
    conditioned_cloud = np.asarray(cloud["conditioned_cloud"], dtype=np.float32)
    tube = cloud["tube"]
    if not isinstance(tube, ProbabilisticTrajectoryTube):
        raise TypeError("cloud['tube'] must be ProbabilisticTrajectoryTube")
    target_xyz = np.asarray(cloud["target_xyz"], dtype=np.float32)
    transitions = np.asarray(cloud["target_transitions"], dtype=np.int32)
    target_gripper = np.asarray(cloud["target_gripper"], dtype=np.float32)
    source_gripper = np.asarray(cloud["source_gripper"], dtype=np.float32)
    progress = np.linspace(0.0, 1.0, target_xyz.shape[0], dtype=np.float32)
    diag = tube_radius_diagnostics(tube)

    ax_cloud3d = fig.add_subplot(grid[row, 0], projection="3d")
    _add_tube_surface(ax_cloud3d, tube, stride=max(1, target_xyz.shape[0] // 70))
    for demo in tube.aligned_points:
        _plot_colored_line(ax_cloud3d, demo, progress, alpha=0.12, linewidth=0.6, cmap_name="Greys")
    _plot_colored_line(ax_cloud3d, tube.mean, progress, alpha=0.9, linewidth=1.8, cmap_name="plasma")
    if len(transitions):
        event_xyz = target_xyz[np.clip(transitions, 0, len(target_xyz) - 1)]
        ax_cloud3d.scatter(event_xyz[:, 0], event_xyz[:, 1], event_xyz[:, 2], color="#d62728", s=42)
    ax_cloud3d.set_title(f"{title_prefix}: probabilistic tube surface")
    ax_cloud3d.set_xlabel("x")
    ax_cloud3d.set_ylabel("y")
    ax_cloud3d.set_zlabel("z")
    if view is not None:
        ax_cloud3d.view_init(elev=view[0], azim=view[1])
    _axis_equal_3d(ax_cloud3d, np.concatenate([tube.aligned_points.reshape(-1, 3), tube.mean], axis=0))

    ax_flow = fig.add_subplot(grid[row, 1])
    num_sections = min(9, max(4, target_xyz.shape[0] // 18))
    section_indices = np.linspace(0, target_xyz.shape[0] - 2, num_sections).round().astype(np.int32)
    normal_offsets = []
    for t_idx in range(target_xyz.shape[0]):
        normal_offsets.append((tube.aligned_points[:, t_idx, :] - tube.mean[t_idx]) @ tube.frame[t_idx, 1:].T)
    normal_offsets = np.stack(normal_offsets, axis=1)
    scale = float(np.percentile(np.linalg.norm(normal_offsets.reshape(-1, 2), axis=-1), 90))
    if scale <= 1e-6:
        scale = float(np.percentile(np.linalg.norm(np.diff(tube.aligned_points, axis=1).reshape(-1, 3), axis=-1), 90))
    if scale <= 1e-6:
        scale = 0.01
    spacing = scale * 5.0
    plotted_xy = []
    for section_col, t_idx in enumerate(section_indices):
        center_x = section_col * spacing
        points = normal_offsets[:, t_idx, :]
        velocity = tube.aligned_points[:, t_idx + 1, :] - tube.aligned_points[:, t_idx, :]
        arrows = velocity @ tube.frame[t_idx, 1:].T
        arrow_norm = np.linalg.norm(arrows, axis=-1)
        if float(np.max(arrow_norm)) > 1e-8:
            arrows = arrows / (float(np.percentile(arrow_norm, 90)) + 1e-8) * scale * 0.9
        x = center_x + points[:, 0]
        y = points[:, 1]
        color = colormaps["viridis"](progress[t_idx])
        ellipse1 = _ellipse_points_from_cov(np.array([center_x, 0.0], dtype=np.float32), tube.cov2[t_idx], radius=1.0)
        ellipse2 = _ellipse_points_from_cov(np.array([center_x, 0.0], dtype=np.float32), tube.cov2[t_idx], radius=2.0)
        ax_flow.fill(ellipse2[:, 0], ellipse2[:, 1], color=color, alpha=0.08, linewidth=0.0)
        ax_flow.plot(ellipse1[:, 0], ellipse1[:, 1], color=color, alpha=0.75, linewidth=1.0)
        ax_flow.plot(ellipse2[:, 0], ellipse2[:, 1], color=color, alpha=0.38, linewidth=0.8)
        ax_flow.scatter(x, y, color=color, alpha=0.42, s=14)
        ax_flow.quiver(
            x,
            y,
            arrows[:, 0],
            arrows[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=0.7,
            width=0.003,
        )
        ax_flow.axvline(center_x, color="#d0d7de", linewidth=0.7, alpha=0.7)
        if bool(diag["blackhole_mask"][t_idx]):
            ax_flow.axvline(center_x, color="#d62728", linewidth=1.4, alpha=0.85)
        elif bool(diag["suspicious_non_event_minima"][t_idx]):
            ax_flow.axvline(center_x, color="#ff7f0e", linewidth=1.4, alpha=0.9)
            ax_flow.scatter([center_x], [0.0], marker="x", color="#ff7f0e", s=42)
        ax_flow.text(center_x, -scale * 2.3, f"{progress[t_idx]:.2f}", ha="center", va="top", fontsize=7)
        plotted_xy.append(np.stack([x, y], axis=-1))
    transition_set = set(int(x) for x in transitions.tolist())
    for section_col, t_idx in enumerate(section_indices):
        near_event = min((abs(int(t_idx) - event) for event in transition_set), default=999)
        if near_event <= 2:
            ax_flow.axvline(section_col * spacing, color="#d62728", linewidth=1.3, alpha=0.8)
    ax_flow.set_title("phase-separated probabilistic cross-section flow")
    ax_flow.set_xlabel("separated phase sections")
    ax_flow.set_ylabel("local z offset")
    ax_flow.grid(True, alpha=0.18)
    if plotted_xy:
        _axis_equal_2d(ax_flow, np.concatenate(plotted_xy, axis=0), pad=0.25)

    ax_phase = fig.add_subplot(grid[row, 2], projection="3d")
    for demo in base_cloud:
        _plot_colored_line(ax_phase, demo, progress, alpha=0.2, linewidth=0.8, cmap_name="Greys")
    _plot_colored_line(ax_phase, target_xyz, progress, alpha=1.0, linewidth=3.0, cmap_name="plasma")
    for transition in transitions:
        event = target_xyz[int(np.clip(transition, 0, len(target_xyz) - 1))]
        ax_phase.scatter(event[0], event[1], event[2], color="#d62728", s=34)
    ax_phase.set_title("target demo phase trajectory")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.set_zlabel("z")
    if view is not None:
        ax_phase.view_init(elev=view[0], azim=view[1])
    _axis_equal_3d(ax_phase, base_cloud.reshape(-1, 3))


def render_demo_conditioned_comparison(
    clouds: list[dict[str, np.ndarray | list[str] | int | str]],
    output: Path,
    *,
    view: tuple[float, float] | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as exc:
        raise SystemExit("Install visualization deps: uv sync --extra viz") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = len(clouds)
    fig = plt.figure(figsize=(18, 5.4 * rows), constrained_layout=True)
    grid = GridSpec(rows, 3, figure=fig, wspace=0.18, hspace=0.24)
    for row, cloud in enumerate(clouds):
        _render_cloud_row(
            fig,
            grid,
            row,
            cloud,
            title_prefix=f"target demo {row}",
            view=view,
        )
    fig.suptitle("Demo-conditioned PATCS cloud check", fontsize=16)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a demo-conditioned PATCS full-trajectory cloud.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True, help="Task id like task_02")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--compare-targets", type=int, default=1)
    parser.add_argument("--max-demos", type=int, default=12)
    parser.add_argument("--dims", nargs=3, type=int, default=(0, 1, 2))
    parser.add_argument("--event-window", type=int, default=3)
    parser.add_argument("--view", nargs=2, type=float)
    args = parser.parse_args()

    records = _load_records_from_manifest(
        args.manifest,
        suite=args.suite,
        task=args.task,
        max_demos=args.max_demos,
    )
    target_indices = list(range(args.target_index, min(len(records), args.target_index + max(1, args.compare_targets))))
    clouds = [
        build_demo_conditioned_cloud(
            records,
            target_index=target_index,
            dims=tuple(args.dims),
            event_window=args.event_window,
        )
        for target_index in target_indices
    ]
    view = None if args.view is None else (args.view[0], args.view[1])
    if len(clouds) == 1:
        render_demo_conditioned_comparison(clouds, args.output, view=view)
    else:
        render_demo_conditioned_comparison(clouds, args.output, view=view)

    npz_path = args.output.with_suffix(".npz")
    arrays = {}
    for row, cloud in enumerate(clouds):
        prefix = f"demo{row}_"
        for key in (
            "base_cloud",
            "conditioned_cloud",
            "target_xyz",
            "event_mask",
            "target_transitions",
            "target_gripper",
            "source_gripper",
            "source_lengths",
            "source_transition_counts",
        ):
            arrays[prefix + key] = cloud[key]
        tube = cloud["tube"]
        if isinstance(tube, ProbabilisticTrajectoryTube):
            diag = tube_radius_diagnostics(tube)
            arrays[prefix + "tube_mean"] = tube.mean
            arrays[prefix + "tube_frame"] = tube.frame
            arrays[prefix + "tube_cov2"] = tube.cov2
            arrays[prefix + "tube_base_cov2"] = tube.base_cov2
            arrays[prefix + "tube_olive_scale"] = tube.olive_scale
            arrays[prefix + "tube_event_scale"] = tube.event_scale
            arrays[prefix + "tube_blackhole_mask"] = tube.blackhole_mask
            arrays[prefix + "tube_radius_minor"] = diag["radius_minor"]
            arrays[prefix + "tube_radius_major"] = diag["radius_major"]
            arrays[prefix + "tube_radius_area"] = diag["radius_area"]
            arrays[prefix + "tube_nearest_transition_distance"] = diag["nearest_transition_distance"]
            arrays[prefix + "tube_suspicious_non_event_minima"] = diag["suspicious_non_event_minima"]
    np.savez_compressed(npz_path, **arrays)
    summary = {
        "output": str(args.output),
        "npz": str(npz_path),
        "compare_targets": len(clouds),
        "event_window": int(args.event_window),
        "targets": [
            {
                "base_cloud_shape": list(np.asarray(cloud["base_cloud"]).shape),
                "conditioned_cloud_shape": list(np.asarray(cloud["conditioned_cloud"]).shape),
                "target_xyz_shape": list(np.asarray(cloud["target_xyz"]).shape),
                "target_transition_count": int(len(np.asarray(cloud["target_transitions"]))),
                "tube_mean_shape": (
                    list(cloud["tube"].mean.shape)
                    if isinstance(cloud["tube"], ProbabilisticTrajectoryTube)
                    else None
                ),
                "tube_cov2_shape": (
                    list(cloud["tube"].cov2.shape)
                    if isinstance(cloud["tube"], ProbabilisticTrajectoryTube)
                    else None
                ),
                "event_scale_min": (
                    float(np.min(cloud["tube"].event_scale))
                    if isinstance(cloud["tube"], ProbabilisticTrajectoryTube)
                    else None
                ),
                "olive_scale_min": (
                    float(np.min(cloud["tube"].olive_scale))
                    if isinstance(cloud["tube"], ProbabilisticTrajectoryTube)
                    else None
                ),
                "blackhole_count": (
                    int(np.sum(cloud["tube"].blackhole_mask))
                    if isinstance(cloud["tube"], ProbabilisticTrajectoryTube)
                    else None
                ),
                "suspicious_non_event_minima_count": (
                    int(np.sum(tube_radius_diagnostics(cloud["tube"])["suspicious_non_event_minima"]))
                    if isinstance(cloud["tube"], ProbabilisticTrajectoryTube)
                    else None
                ),
                "source_lengths": np.asarray(cloud["source_lengths"]).tolist(),
                "source_transition_counts": np.asarray(cloud["source_transition_counts"]).tolist(),
                "target_path": cloud["target_path"],
                "target_language": cloud["target_language"],
            }
            for cloud in clouds
        ],
        "logic": (
            "full trajectories are resampled to the target demo length; a Gaussian normal-plane "
            "probability tube is built around the target-conditioned distribution; event windows "
            "use black-hole contraction toward the target demo gripper-transition observations"
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
