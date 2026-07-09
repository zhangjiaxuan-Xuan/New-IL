from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from new_il.patcs_probabilistic import (
    ProbabilisticTubeConfig,
    build_probabilistic_trajectory_tube,
    tube_radius_diagnostics,
    tube_surface_mesh,
)


def _plot_colored_line(ax, xyz: np.ndarray, values: np.ndarray, *, alpha: float, linewidth: float, cmap_name: str) -> None:
    cmap = colormaps[cmap_name]
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    collection = Line3DCollection(segments, colors=cmap(values[:-1]), linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(collection)


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


def _add_tube_surface(ax, tube, *, stride: int) -> None:
    vertices, faces, density = tube_surface_mesh(tube, stride=stride)
    if len(faces) == 0:
        return
    face_sections = np.arange(len(faces), dtype=np.int32) // (2 * tube.config.surface_sides)
    section_density = density[np.clip(face_sections, 0, len(density) - 1)]
    density_norm = (section_density - section_density.min()) / max(float(np.ptp(section_density)), 1e-8)
    colors = colormaps["viridis"](density_norm)
    colors[:, 3] = 0.16 + 0.28 * density_norm
    mesh = Poly3DCollection(vertices[faces], facecolors=colors, edgecolors="none", linewidths=0.0)
    ax.add_collection3d(mesh)


def _global_gaussian_density(points: np.ndarray, query: np.ndarray, *, min_std: float) -> np.ndarray:
    mean = points.mean(axis=0)
    cov = np.cov(points - mean, rowvar=False, bias=False).astype(np.float32)
    cov += np.eye(3, dtype=np.float32) * float(min_std ** 2)
    inv = np.linalg.inv(cov)
    diff = query - mean[None, :]
    mahal = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return np.exp(-0.5 * mahal).astype(np.float32, copy=False)


def _add_global_density_points(
    ax,
    points: np.ndarray,
    *,
    min_std: float,
    grid_size: int = 34,
    quantile: float = 0.82,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.12, min_std * 3.0)
    axes = [np.linspace(mins[d] - pad[d], maxs[d] + pad[d], grid_size, dtype=np.float32) for d in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    query = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)
    density = _global_gaussian_density(points, query, min_std=min_std)
    threshold = float(np.quantile(density, quantile))
    mask = density >= threshold
    selected = query[mask]
    selected_density = density[mask]
    order = np.argsort(selected_density)
    selected = selected[order]
    selected_density = selected_density[order]
    norm = (selected_density - selected_density.min()) / max(float(np.ptp(selected_density)), 1e-8)
    colors = colormaps["viridis"](norm)
    colors[:, 3] = 0.035 + 0.22 * norm
    ax.scatter(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        s=10,
        c=colors,
        marker="o",
        depthshade=False,
        linewidths=0.0,
    )
    return selected


def build_synthetic_line_demos(
    *,
    num_demos: int,
    length: int,
    radius: float,
    wobble: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 1.0, length, dtype=np.float32)
    center = np.stack([phase, np.zeros_like(phase), np.zeros_like(phase)], axis=-1)
    demos = []
    angles = np.linspace(0.0, 2.0 * np.pi, num_demos, endpoint=False, dtype=np.float32)
    for idx, angle in enumerate(angles):
        demo_radius = radius * (0.65 + 0.55 * (idx + 1) / max(num_demos, 1))
        normal = np.stack(
            [
                np.zeros_like(phase),
                np.cos(angle) * np.ones_like(phase),
                np.sin(angle) * np.ones_like(phase),
            ],
            axis=-1,
        )
        side = np.stack(
            [
                np.zeros_like(phase),
                -np.sin(angle) * np.ones_like(phase),
                np.cos(angle) * np.ones_like(phase),
            ],
            axis=-1,
        )
        demo = center + demo_radius * normal
        demo += wobble * np.sin(phase[:, None] * np.pi * 2.0 + angle) * side
        demo += rng.normal(0.0, radius * 0.03, size=demo.shape).astype(np.float32)
        demos.append(demo.astype(np.float32, copy=False))
    return demos


def build_synthetic_pose_demos(
    *,
    num_demos: int,
    length: int,
    radius: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 101)
    phase = np.linspace(0.0, 1.0, length, dtype=np.float32)
    base = np.stack(
        [
            0.15 * np.sin(np.pi * phase),
            0.55 * phase,
            0.12 * np.cos(np.pi * phase),
        ],
        axis=-1,
    )
    demos = []
    angles = np.linspace(0.0, 2.0 * np.pi, num_demos, endpoint=False, dtype=np.float32)
    for idx, angle in enumerate(angles):
        spread = radius * (0.7 + 0.4 * (idx + 1) / max(num_demos, 1))
        offset = np.stack(
            [
                np.sin(angle) * np.ones_like(phase),
                np.zeros_like(phase),
                np.cos(angle) * np.ones_like(phase),
            ],
            axis=-1,
        )
        demo = base + spread * offset
        demo += rng.normal(0.0, radius * 0.025, size=demo.shape).astype(np.float32)
        demos.append(demo.astype(np.float32, copy=False))
    return demos


def build_synthetic_gripper(length: int, transitions: np.ndarray) -> np.ndarray:
    gripper = np.zeros((length,), dtype=np.float32)
    state = 0.0
    transition_set = set(int(x) for x in transitions.tolist())
    for t in range(length):
        if t in transition_set:
            state = 1.0 - state
        gripper[t] = state
    return gripper


def render_synthetic_tube(
    output: Path,
    *,
    num_demos: int,
    length: int,
    radius: float,
    wobble: float,
    transitions: np.ndarray,
    config: ProbabilisticTubeConfig,
    pose_radius: float,
    seed: int,
    view: tuple[float, float] | None,
) -> dict:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install visualization deps: uv sync --extra viz") from exc

    demos = build_synthetic_line_demos(
        num_demos=num_demos,
        length=length,
        radius=radius,
        wobble=wobble,
        seed=seed,
    )
    tube = build_probabilistic_trajectory_tube(
        demos,
        target_index=0,
        transitions=transitions,
        config=config,
    )
    pose_demos = build_synthetic_pose_demos(
        num_demos=num_demos,
        length=length,
        radius=pose_radius,
        seed=seed,
    )
    pose_tube = build_probabilistic_trajectory_tube(
        pose_demos,
        target_index=0,
        transitions=transitions,
        config=config,
    )
    gripper = build_synthetic_gripper(length, transitions)
    diag = tube_radius_diagnostics(tube)
    pose_diag = tube_radius_diagnostics(pose_tube)
    progress = np.linspace(0.0, 1.0, len(tube.target_xyz), dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    if config.density_mode == "global_3d_gaussian":
        xyz_for_limits = _add_global_density_points(
            ax3d,
            tube.aligned_points,
            min_std=config.min_std,
        )
    else:
        _add_tube_surface(ax3d, tube, stride=1)
        xyz_for_limits = tube_surface_mesh(tube, stride=1)[0]
    for demo in tube.aligned_points:
        _plot_colored_line(ax3d, demo, progress, alpha=0.16, linewidth=0.8, cmap_name="Greys")
    _plot_colored_line(ax3d, tube.mean, progress, alpha=0.95, linewidth=2.2, cmap_name="plasma")
    event_xyz = tube.target_xyz[np.clip(tube.transitions, 0, len(tube.target_xyz) - 1)]
    if len(event_xyz):
        ax3d.scatter(event_xyz[:, 0], event_xyz[:, 1], event_xyz[:, 2], color="#d62728", s=42)
    ax3d.set_title(f"xyz {config.density_mode} density")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    if view is not None:
        ax3d.view_init(elev=view[0], azim=view[1])
    _axis_equal_3d(ax3d, xyz_for_limits)

    ax_pose = fig.add_subplot(2, 3, 2, projection="3d")
    if config.density_mode == "global_3d_gaussian":
        pose_for_limits = _add_global_density_points(
            ax_pose,
            pose_tube.aligned_points,
            min_std=config.min_std,
        )
    else:
        _add_tube_surface(ax_pose, pose_tube, stride=1)
        pose_for_limits = tube_surface_mesh(pose_tube, stride=1)[0]
    for demo in pose_tube.aligned_points:
        _plot_colored_line(ax_pose, demo, progress, alpha=0.16, linewidth=0.8, cmap_name="Greys")
    _plot_colored_line(ax_pose, pose_tube.mean, progress, alpha=0.95, linewidth=2.2, cmap_name="plasma")
    pose_event = pose_tube.target_xyz[np.clip(pose_tube.transitions, 0, len(pose_tube.target_xyz) - 1)]
    if len(pose_event):
        ax_pose.scatter(pose_event[:, 0], pose_event[:, 1], pose_event[:, 2], color="#d62728", s=42)
    ax_pose.set_title(f"pose {config.density_mode} density")
    ax_pose.set_xlabel("roll-like")
    ax_pose.set_ylabel("pitch-like")
    ax_pose.set_zlabel("yaw-like")
    if view is not None:
        ax_pose.view_init(elev=view[0], azim=view[1])
    _axis_equal_3d(ax_pose, pose_for_limits)

    ax_gripper = fig.add_subplot(2, 3, 3)
    t = np.arange(length)
    ax_gripper.step(t, gripper, where="post", color="#1f77b4", linewidth=2.2, label="hard target")
    for transition in transitions:
        ax_gripper.axvline(int(transition), color="#d62728", linewidth=1.2, alpha=0.85)
    ax_gripper.set_title("gripper hard supervision")
    ax_gripper.set_xlabel("phase index")
    ax_gripper.set_ylabel("state")
    ax_gripper.set_ylim(-0.15, 1.15)
    ax_gripper.grid(True, alpha=0.25)
    ax_gripper.legend(loc="best")

    ax_radius = fig.add_subplot(2, 3, 4)
    t = np.arange(len(tube.target_xyz))
    ax_radius.plot(t, diag["radius_minor"], label="minor radius", color="#1f77b4")
    ax_radius.plot(t, diag["radius_major"], label="major radius", color="#2ca02c")
    ax_radius.plot(t, diag["olive_scale"] * max(float(diag["radius_major"].max()), 1e-8), label="olive scale", color="#9467bd", alpha=0.7)
    ax_radius.plot(t, diag["channel_scale"] * max(float(diag["radius_major"].max()), 1e-8), label="channel scale", color="#ff7f0e", alpha=0.75)
    for transition in transitions:
        ax_radius.axvline(int(transition), color="#d62728", linewidth=1.0, alpha=0.85)
    for idx in np.flatnonzero(diag["blackhole_mask"]):
        ax_radius.axvspan(idx - 0.5, idx + 0.5, color="#d62728", alpha=0.06)
    ax_radius.set_title("radius / olive / black-hole")
    ax_radius.set_xlabel("phase index")
    ax_radius.grid(True, alpha=0.25)
    ax_radius.legend(loc="best")

    ax_pose_radius = fig.add_subplot(2, 3, 5)
    ax_pose_radius.plot(t, pose_diag["radius_minor"], label="pose minor radius", color="#1f77b4")
    ax_pose_radius.plot(t, pose_diag["radius_major"], label="pose major radius", color="#2ca02c")
    ax_pose_radius.plot(
        t,
        pose_diag["olive_scale"] * max(float(pose_diag["radius_major"].max()), 1e-8),
        label="olive scale",
        color="#9467bd",
        alpha=0.7,
    )
    ax_pose_radius.plot(
        t,
        pose_diag["channel_scale"] * max(float(pose_diag["radius_major"].max()), 1e-8),
        label="channel scale",
        color="#ff7f0e",
        alpha=0.75,
    )
    for transition in transitions:
        ax_pose_radius.axvline(int(transition), color="#d62728", linewidth=1.0, alpha=0.85)
    for idx in np.flatnonzero(pose_diag["blackhole_mask"]):
        ax_pose_radius.axvspan(idx - 0.5, idx + 0.5, color="#d62728", alpha=0.06)
    ax_pose_radius.set_title("pose radius / olive / black-hole")
    ax_pose_radius.set_xlabel("phase index")
    ax_pose_radius.grid(True, alpha=0.25)
    ax_pose_radius.legend(loc="best")

    ax_cross = fig.add_subplot(2, 3, 6)
    sample_idx = np.linspace(0, len(tube.target_xyz) - 1, 9).round().astype(np.int32)
    spacing = max(float(diag["radius_major"].max()) * 4.5, radius * 1.8)
    for col, idx in enumerate(sample_idx):
        center = col * spacing
        cov = tube.cov2[int(idx)]
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-12)
        angles = np.linspace(0.0, 2.0 * np.pi, 80)
        unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        for sigma, alpha, lw in ((1.0, 0.7, 1.1), (2.0, 0.25, 0.8)):
            ellipse = np.array([center, 0.0]) + (unit @ vecs.T) * (np.sqrt(vals)[None, :] * sigma)
            ax_cross.plot(ellipse[:, 0], ellipse[:, 1], color=colormaps["viridis"](progress[idx]), alpha=alpha, linewidth=lw)
        if diag["blackhole_mask"][idx]:
            ax_cross.axvline(center, color="#d62728", linewidth=1.2, alpha=0.8)
        ax_cross.text(center, -spacing * 0.22, f"{progress[idx]:.2f}", ha="center", va="top", fontsize=7)
    ax_cross.set_title("xyz probability cross-sections")
    ax_cross.set_xlabel("phase sections")
    ax_cross.set_ylabel("local normal offset")
    ax_cross.set_aspect("equal", adjustable="box")
    ax_cross.grid(True, alpha=0.2)

    fig.savefig(output, dpi=180)
    plt.close(fig)

    npz_path = output.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        aligned_points=tube.aligned_points,
        target_xyz=tube.target_xyz,
        mean=tube.mean,
        cov2=tube.cov2,
        base_cov2=tube.base_cov2,
        pose_aligned_points=pose_tube.aligned_points,
        pose_target=pose_tube.target_xyz,
        pose_mean=pose_tube.mean,
        pose_cov2=pose_tube.cov2,
        gripper=gripper,
        olive_scale=tube.olive_scale,
        channel_scale=tube.channel_scale,
        pose_olive_scale=pose_tube.olive_scale,
        pose_channel_scale=pose_tube.channel_scale,
        event_scale=tube.event_scale,
        pose_event_scale=pose_tube.event_scale,
        blackhole_mask=tube.blackhole_mask,
        pose_blackhole_mask=pose_tube.blackhole_mask,
        transitions=tube.transitions,
        radius_minor=diag["radius_minor"],
        radius_major=diag["radius_major"],
        radius_area=diag["radius_area"],
        pose_radius_minor=pose_diag["radius_minor"],
        pose_radius_major=pose_diag["radius_major"],
        pose_radius_area=pose_diag["radius_area"],
        suspicious_non_event_minima=diag["suspicious_non_event_minima"],
        pose_suspicious_non_event_minima=pose_diag["suspicious_non_event_minima"],
    )
    summary = {
        "output": str(output),
        "npz": str(npz_path),
        "num_demos": num_demos,
        "length": length,
        "radius": radius,
        "pose_radius": pose_radius,
        "wobble": wobble,
        "transitions": transitions.tolist(),
        "config": config.__dict__,
        "density_mode": config.density_mode,
        "suspicious_non_event_minima_count": int(np.sum(diag["suspicious_non_event_minima"])),
        "pose_suspicious_non_event_minima_count": int(np.sum(pose_diag["suspicious_non_event_minima"])),
        "blackhole_count": int(np.sum(diag["blackhole_mask"])),
        "pose_blackhole_count": int(np.sum(pose_diag["blackhole_mask"])),
        "radius_minor_minmax": [float(diag["radius_minor"].min()), float(diag["radius_minor"].max())],
        "radius_major_minmax": [float(diag["radius_major"].min()), float(diag["radius_major"].max())],
        "pose_radius_minor_minmax": [float(pose_diag["radius_minor"].min()), float(pose_diag["radius_minor"].max())],
        "pose_radius_major_minmax": [float(pose_diag["radius_major"].min()), float(pose_diag["radius_major"].max())],
        "supervision_schema": "xyz probability cloud + pose probability cloud + gripper hard supervision",
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a synthetic straight-line PATCS probability tube.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-demos", type=int, default=16)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--pose-radius", type=float, default=0.05)
    parser.add_argument("--wobble", type=float, default=0.012)
    parser.add_argument("--transitions", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--event-window", type=int, default=8)
    parser.add_argument("--blackhole-window", type=int, default=2)
    parser.add_argument("--density-mode", choices=["phase_2d", "global_3d_gaussian"], default="phase_2d")
    parser.add_argument("--precision-channel-window", type=int, default=12)
    parser.add_argument("--precision-channel-radius", type=float, default=0.018)
    parser.add_argument("--event-radius", type=float, default=0.006)
    parser.add_argument("--min-envelope-std", type=float, default=0.02)
    parser.add_argument("--olive-power", type=float, default=0.75)
    parser.add_argument("--radius-smooth-window", type=int, default=9)
    parser.add_argument("--max-radius-ratio", type=float, default=1.2)
    parser.add_argument("--target-mean-blend", type=float, default=0.35)
    parser.add_argument("--iso-sigma", type=float, default=2.0)
    parser.add_argument("--surface-sides", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--view", nargs=2, type=float, default=(22.0, -68.0))
    args = parser.parse_args()
    config = ProbabilisticTubeConfig(
        density_mode=args.density_mode,
        event_window=args.event_window,
        blackhole_window=args.blackhole_window,
        precision_channel_window=args.precision_channel_window,
        precision_channel_radius=args.precision_channel_radius,
        event_radius=args.event_radius,
        min_envelope_std=args.min_envelope_std,
        olive_power=args.olive_power,
        radius_smooth_window=args.radius_smooth_window,
        max_radius_ratio=args.max_radius_ratio,
        target_mean_blend=args.target_mean_blend,
        iso_sigma=args.iso_sigma,
        surface_sides=args.surface_sides,
    )
    summary = render_synthetic_tube(
        args.output,
        num_demos=args.num_demos,
        length=args.length,
        radius=args.radius,
        wobble=args.wobble,
        transitions=np.asarray(args.transitions, dtype=np.int32),
        config=config,
        pose_radius=args.pose_radius,
        seed=args.seed,
        view=None if args.view is None else (args.view[0], args.view[1]),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
