from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from new_il.patcs import gripper_transition_indices, segment_boundaries


REQUIRED_KEYS = {
    "actions",
    "success",
    "metadata_json",
    "observation.state",
}


@dataclass(frozen=True)
class RolloutRecord:
    source_path: str
    grouped_path: str | None
    task_suite_name: str
    task_idx: int
    episode_idx: int | None
    round_idx: int | None
    seed: int | None
    success: bool
    length: int
    action_mean_abs: float
    action_std: float
    action_min: float
    action_max: float
    xyz_norm_mean: float
    repeated_action_rate: float
    all_zero: bool
    gripper_transition_count: int
    stage_count: int
    state_shape: list[int]
    image_shape: list[int] | None
    image2_shape: list[int] | None
    has_image: bool
    has_image2: bool
    language: str | None


@dataclass(frozen=True)
class BadRollout:
    source_path: str
    error_type: str
    error: str


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_scalar_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _shape_or_none(data: np.lib.npyio.NpzFile, key: str) -> list[int] | None:
    if key not in data.files:
        return None
    return [int(x) for x in data[key].shape]


def _language_or_none(data: np.lib.npyio.NpzFile) -> str | None:
    if "language" not in data.files:
        return None
    value = data["language"]
    if value.shape == ():
        return str(value.item())
    return str(value)


def _read_record(
    path: Path,
    grouped_path: Path | None = None,
    *,
    inspect_images: bool = False,
) -> RolloutRecord:
    data = np.load(path, allow_pickle=False)
    missing = REQUIRED_KEYS - set(data.files)
    if missing:
        raise KeyError(f"missing required arrays: {sorted(missing)}")

    metadata = json.loads(str(data["metadata_json"]))
    task_suite_name = str(metadata["task_suite_name"])
    task_idx = int(metadata["task_idx"])
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"actions must be [T, 7], got {actions.shape}")
    state = np.asarray(data["observation.state"], dtype=np.float32)
    if state.shape[0] != actions.shape[0]:
        raise ValueError(f"state/actions length mismatch: {state.shape[0]} vs {actions.shape[0]}")

    repeated = (
        np.all(np.isclose(np.diff(actions, axis=0), 0.0, atol=1e-6), axis=1)
        if len(actions) > 1
        else np.asarray([], dtype=bool)
    )
    success = bool(data["success"])
    transitions = gripper_transition_indices(actions, gripper_index=-1, threshold=0.0)
    return RolloutRecord(
        source_path=str(path),
        grouped_path=str(grouped_path) if grouped_path is not None else None,
        task_suite_name=task_suite_name,
        task_idx=task_idx,
        episode_idx=_as_scalar_int(metadata.get("episode_idx")),
        round_idx=_as_scalar_int(metadata.get("round_idx")),
        seed=_as_scalar_int(metadata.get("seed")),
        success=success,
        length=int(actions.shape[0]),
        action_mean_abs=float(np.abs(actions).mean()),
        action_std=float(actions.std()),
        action_min=float(actions.min()),
        action_max=float(actions.max()),
        xyz_norm_mean=float(np.linalg.norm(actions[:, :3], axis=1).mean()),
        repeated_action_rate=float(repeated.astype(np.float32).mean()) if repeated.size else 0.0,
        all_zero=bool(np.allclose(actions, 0.0, atol=1e-8)),
        gripper_transition_count=len(transitions),
        stage_count=len(segment_boundaries(len(actions), transitions)),
        state_shape=[int(x) for x in state.shape],
        image_shape=_shape_or_none(data, "observation.images.image") if inspect_images else None,
        image2_shape=_shape_or_none(data, "observation.images.image2") if inspect_images else None,
        has_image="observation.images.image" in data.files,
        has_image2="observation.images.image2" in data.files,
        language=_language_or_none(data),
    )


def _stat(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _group_key(record: RolloutRecord) -> str:
    return f"{record.task_suite_name}/task_{record.task_idx:02d}"


def _safe_link_name(path: Path) -> str:
    parent = path.parents[1].name if len(path.parents) > 1 else "rollout"
    return f"{parent}__{path.name}"


def _make_grouped_link(source: Path, output_dir: Path, record: RolloutRecord) -> Path:
    status_dir = "success_rollouts" if record.success else "failed_rollouts"
    target_dir = output_dir / "by_task" / record.task_suite_name / f"task_{record.task_idx:02d}" / status_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    link_path = target_dir / _safe_link_name(source)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(source.resolve())
    return link_path


def _summarize(records: list[RolloutRecord], bad: list[BadRollout]) -> dict[str, Any]:
    groups: dict[str, list[RolloutRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)

    group_summary = {}
    for key, items in sorted(groups.items()):
        success_items = [item for item in items if item.success]
        group_summary[key] = {
            "total": len(items),
            "success": len(success_items),
            "failed": len(items) - len(success_items),
            "stage_counts_success": {
                str(stage_count): sum(1 for item in success_items if item.stage_count == stage_count)
                for stage_count in sorted({item.stage_count for item in success_items})
            },
            "length": _stat([float(item.length) for item in success_items]),
            "xyz_norm_mean": _stat([item.xyz_norm_mean for item in success_items]),
            "action_mean_abs": _stat([item.action_mean_abs for item in success_items]),
            "all_zero_success": sum(1 for item in success_items if item.all_zero),
            "high_repeated_success": sum(1 for item in success_items if item.repeated_action_rate > 0.01),
        }

    success_records = [record for record in records if record.success]
    return {
        "total_npz_readable": len(records),
        "total_bad_npz": len(bad),
        "success": len(success_records),
        "failed": len(records) - len(success_records),
        "groups": group_summary,
        "global_success_stats": {
            "length": _stat([float(item.length) for item in success_records]),
            "xyz_norm_mean": _stat([item.xyz_norm_mean for item in success_records]),
            "action_mean_abs": _stat([item.action_mean_abs for item in success_records]),
            "action_std": _stat([item.action_std for item in success_records]),
            "repeated_action_rate": _stat([item.repeated_action_rate for item in success_records]),
            "all_zero_success": sum(1 for item in success_records if item.all_zero),
            "high_repeated_success": sum(
                1 for item in success_records if item.repeated_action_rate > 0.01
            ),
        },
        "bad_rollouts": [asdict(item) for item in bad],
    }


def build_rollout_manifest(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    create_links: bool = True,
    include_failed_links: bool = False,
    inspect_images: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[RolloutRecord] = []
    bad: list[BadRollout] = []
    for path in sorted(input_dir.glob("**/*.npz")):
        try:
            first_record = _read_record(path, inspect_images=inspect_images)
            grouped_path = None
            if create_links and (first_record.success or include_failed_links):
                grouped_path = _make_grouped_link(path, output_dir, first_record)
            records.append(_read_record(path, grouped_path, inspect_images=inspect_images))
        except Exception as exc:
            bad.append(BadRollout(str(path), type(exc).__name__, str(exc)))

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "create_links": create_links,
        "include_failed_links": include_failed_links,
        "inspect_images": inspect_images,
        "records": [asdict(record) for record in records],
        "bad_rollouts": [asdict(item) for item in bad],
    }
    summary = _summarize(records, bad)
    _json_dump(output_dir / "manifest.json", manifest)
    _json_dump(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate rollout NPZ files and group them by LIBERO suite/task."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw rollout collection directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest/grouping directory.")
    parser.add_argument("--no-links", action="store_true", help="Only write JSON, do not create symlinks.")
    parser.add_argument(
        "--include-failed-links",
        action="store_true",
        help="Also link failed rollout NPZ files into by_task/*/failed_rollouts.",
    )
    parser.add_argument(
        "--inspect-images",
        action="store_true",
        help="Read image arrays to record exact image shapes. Slower for compressed rollout NPZ files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory first.")
    args = parser.parse_args()

    summary = build_rollout_manifest(
        args.input,
        args.output,
        create_links=not args.no_links,
        include_failed_links=args.include_failed_links,
        inspect_images=args.inspect_images,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
