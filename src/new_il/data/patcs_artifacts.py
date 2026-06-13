from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from new_il.patcs import gripper_transition_indices, normalized_phase, resample_segment, segment_boundaries


@dataclass(frozen=True)
class PatcsArtifactConfig:
    num_demos: int = 16
    num_phase: int = 64
    anchor_demo_index: int = 0
    gripper_index: int = -1
    gripper_threshold: float = 0.0
    margin: float = 0.012
    event_radius: float = 1e-4
    obs_key: str = "ee_pos"
    avoid_event_points: bool = False
    stage_strategy: str = "filter_majority"


def _demo_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def _read_demo_group(handle: h5py.File, name: str, obs_key: str) -> tuple[np.ndarray, np.ndarray]:
    demo = handle["data"][name]
    if obs_key not in demo["obs"]:
        raise KeyError(f"obs/{obs_key} not found in {name}; available: {list(demo['obs'].keys())}")
    states = np.asarray(demo["obs"][obs_key], dtype=np.float32)
    actions = np.asarray(demo["actions"], dtype=np.float32)
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"{name}: obs/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}")
    return states, actions


def _scan_demos(hdf5_path: Path, config: PatcsArtifactConfig) -> list[dict[str, Any]]:
    records = []
    with h5py.File(hdf5_path, "r") as handle:
        demo_names = sorted(handle["data"].keys(), key=_demo_sort_key)
        for name in demo_names:
            states, actions = _read_demo_group(handle, name, config.obs_key)
            transitions = gripper_transition_indices(
                actions,
                gripper_index=config.gripper_index,
                threshold=config.gripper_threshold,
            )
            bounds = segment_boundaries(len(actions), transitions)
            records.append(
                {
                    "name": name,
                    "states": states,
                    "actions": actions,
                    "transitions": transitions,
                    "bounds": bounds,
                }
            )
    return records


def _target_stage_count(records: list[dict[str, Any]], strategy: str) -> int:
    if not records:
        raise RuntimeError("No demonstrations found.")
    if strategy == "strict":
        counts = {len(record["bounds"]) for record in records}
        if len(counts) != 1:
            raise ValueError(f"strict stage strategy requires one stage count, got {sorted(counts)}")
        return counts.pop()
    if strategy != "filter_majority":
        raise ValueError(f"Unknown stage_strategy: {strategy}")
    counts: dict[int, int] = {}
    for record in records:
        counts[len(record["bounds"])] = counts.get(len(record["bounds"]), 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _collect_stage_points(
    hdf5_path: Path,
    config: PatcsArtifactConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    records = _scan_demos(hdf5_path, config)
    expected_stages = _target_stage_count(records, config.stage_strategy)
    used: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    stage_segments: list[list[np.ndarray]] = [[] for _ in range(expected_stages)]
    stage_gripper_votes: list[list[list[float]]] = [[] for _ in range(expected_stages)]
    stage_boundaries = []
    raw_lengths = []
    transition_lists = []
    stage_gripper_by_demo = []

    for record in records:
        name = record["name"]
        states = record["states"]
        actions = record["actions"]
        transitions = record["transitions"]
        bounds = record["bounds"]
        if len(bounds) != expected_stages:
            dropped.append(
                {
                    "demo": name,
                    "reason": "stage_count_mismatch",
                    "stage_count": len(bounds),
                    "expected_stage_count": expected_stages,
                    "transitions": [int(index) for index in transitions],
                }
            )
            continue

        per_demo_stages = []
        per_demo_gripper = []
        gripper_binary = (actions[:, config.gripper_index] > config.gripper_threshold).astype(np.float32)
        for start, end in bounds:
            if end - start < 2:
                break
            per_demo_stages.append(
                resample_segment(
                    states[start:end],
                    config.num_phase,
                    avoid_event_points=config.avoid_event_points,
                )
            )
            per_demo_gripper.append([float(gripper_binary[start]), float(gripper_binary[end - 1])])
        if len(per_demo_stages) != expected_stages:
            dropped.append({"demo": name, "reason": "short_stage"})
            continue

        for stage_idx, segment in enumerate(per_demo_stages):
            stage_segments[stage_idx].append(segment)
            stage_gripper_votes[stage_idx].append(per_demo_gripper[stage_idx])
        used.append(
            {
                "demo": name,
                "length": int(len(actions)),
                "transitions": [int(index) for index in transitions],
                "bounds": [[int(start), int(end)] for start, end in bounds],
                "stage_gripper": per_demo_gripper,
            }
        )
        stage_boundaries.append([[int(start), int(end)] for start, end in bounds])
        raw_lengths.append(int(len(actions)))
        transition_lists.append([int(index) for index in transitions])
        stage_gripper_by_demo.append(per_demo_gripper)
        if len(used) >= config.num_demos:
            break

    if not used:
        raise RuntimeError(f"No valid demonstrations found in {hdf5_path}")
    if config.anchor_demo_index >= len(used):
        raise ValueError(f"anchor_demo_index {config.anchor_demo_index} >= used demos {len(used)}")

    phase_points = np.stack([np.stack(stage, axis=0) for stage in stage_segments], axis=0)
    gripper_states = np.array(
        [
            [
                round(float(np.mean(np.asarray(votes)[:, 0]))),
                round(float(np.mean(np.asarray(votes)[:, 1]))),
            ]
            for votes in stage_gripper_votes
        ],
        dtype=np.float32,
    )
    max_transitions = max((len(transitions) for transitions in transition_lists), default=0)
    transition_indices = np.full((len(used), max_transitions), -1, dtype=np.int32)
    transition_counts = np.zeros((len(used),), dtype=np.int32)
    for demo_idx, transitions in enumerate(transition_lists):
        transition_counts[demo_idx] = len(transitions)
        if transitions:
            transition_indices[demo_idx, : len(transitions)] = transitions

    arrays = {
        "stage_boundaries": np.asarray(stage_boundaries, dtype=np.int32),
        "raw_lengths": np.asarray(raw_lengths, dtype=np.int32),
        "transition_indices": transition_indices,
        "transition_counts": transition_counts,
        "stage_gripper_states_by_demo": np.asarray(stage_gripper_by_demo, dtype=np.int8),
        "demo_ids": np.asarray([item["demo"] for item in used], dtype="S64"),
    }
    metadata = {
        "source": str(hdf5_path),
        "stage_strategy": config.stage_strategy,
        "target_stage_count": expected_stages,
        "used_demos": used,
        "dropped_demos": dropped,
        "num_stages": int(phase_points.shape[0]),
        "num_demos": int(phase_points.shape[1]),
        "num_phase": int(phase_points.shape[2]),
        "state_dim": int(phase_points.shape[3]),
    }
    return phase_points.astype(np.float32, copy=False), gripper_states, metadata, arrays


def _convex_hull_equations(points: np.ndarray) -> np.ndarray | None:
    unique = np.unique(points, axis=0)
    if unique.shape[0] <= points.shape[-1]:
        return None
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise RuntimeError("scipy is required to build PATCS hull artifacts.") from exc
    try:
        return ConvexHull(unique).equations.astype(np.float32, copy=False)
    except Exception:
        return None


def _build_padded_hulls(phase_points: np.ndarray, event_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    equations_by_section: list[tuple[tuple[int, int], np.ndarray]] = []
    max_equations = 0
    stages, _, phases, dims = phase_points.shape
    for stage_idx in range(stages):
        for phase_idx in range(phases):
            if event_mask[stage_idx, phase_idx]:
                continue
            equations = _convex_hull_equations(phase_points[stage_idx, :, phase_idx, :])
            if equations is None:
                continue
            equations_by_section.append(((stage_idx, phase_idx), equations))
            max_equations = max(max_equations, equations.shape[0])

    hull_equations = np.zeros((stages, phases, max_equations, dims + 1), dtype=np.float32)
    hull_valid = np.zeros((stages, phases, max_equations), dtype=bool)
    for (stage_idx, phase_idx), equations in equations_by_section:
        count = equations.shape[0]
        hull_equations[stage_idx, phase_idx, :count, :] = equations
        hull_valid[stage_idx, phase_idx, :count] = True
    return hull_equations, hull_valid


def build_patcs_artifact(
    hdf5_path: Path,
    output: Path,
    config: PatcsArtifactConfig,
) -> Path:
    phase_points, gripper_states, metadata, auxiliary = _collect_stage_points(hdf5_path, config)
    phase_grid = normalized_phase(config.num_phase)
    event_mask = np.zeros((phase_points.shape[0], phase_points.shape[2]), dtype=bool)
    event_mask[:, 0] = True
    event_mask[:, -1] = True
    anchor = phase_points[:, config.anchor_demo_index, :, :]
    hull_equations, hull_valid = _build_padded_hulls(phase_points, event_mask)
    hull_equation_counts = hull_valid.sum(axis=-1).astype(np.int32)
    metadata_json = json.dumps({"config": asdict(config), **metadata}, sort_keys=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.array(1, dtype=np.int32),
        phase_points=phase_points,
        anchor=anchor.astype(np.float32, copy=False),
        anchor_points=anchor.astype(np.float32, copy=False),
        phase=phase_grid,
        phase_grid=phase_grid,
        event_mask=event_mask,
        gripper_states=gripper_states,
        hull_equations=hull_equations,
        hull_valid=hull_valid,
        hull_valid_mask=hull_equation_counts > 0,
        hull_equation_counts=hull_equation_counts,
        margin=np.array(config.margin, dtype=np.float32),
        hull_margin=np.array(config.margin, dtype=np.float32),
        event_radius=np.array(config.event_radius, dtype=np.float32),
        anchor_demo_index=np.array(config.anchor_demo_index, dtype=np.int64),
        anchor_demo_indices=np.full((phase_points.shape[0],), config.anchor_demo_index, dtype=np.int32),
        metadata_json=np.asarray(metadata_json),
        **auxiliary,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(output),
        "config": asdict(config),
        "arrays": {
            "phase_points": list(phase_points.shape),
            "anchor": list(anchor.shape),
            "hull_equations": list(hull_equations.shape),
            "hull_valid": list(hull_valid.shape),
            "hull_equation_counts": list(hull_equation_counts.shape),
            "event_mask": list(event_mask.shape),
            **{key: list(value.shape) for key, value in auxiliary.items()},
        },
        **metadata,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline PA-TCS supervision artifacts from LIBERO HDF5 demos.")
    parser.add_argument("--input", type=Path, required=True, help="LIBERO task HDF5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz artifact path.")
    parser.add_argument("--num-demos", type=int, default=16)
    parser.add_argument("--num-phase", type=int, default=64)
    parser.add_argument("--anchor-demo-index", type=int, default=0)
    parser.add_argument("--obs-key", default="ee_pos")
    parser.add_argument("--margin", type=float, default=0.012)
    parser.add_argument("--event-radius", type=float, default=1e-4)
    parser.add_argument("--avoid-event-points", action="store_true")
    parser.add_argument("--stage-strategy", choices=["strict", "filter_majority"], default="filter_majority")
    args = parser.parse_args()

    config = PatcsArtifactConfig(
        num_demos=args.num_demos,
        num_phase=args.num_phase,
        anchor_demo_index=args.anchor_demo_index,
        margin=args.margin,
        event_radius=args.event_radius,
        obs_key=args.obs_key,
        avoid_event_points=args.avoid_event_points,
        stage_strategy=args.stage_strategy,
    )
    artifact = build_patcs_artifact(args.input, args.output, config)
    print(artifact)
