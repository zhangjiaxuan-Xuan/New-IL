"""Convert OpenVLA modified LIBERO RLDS (tfrecord) to per-task HDF5 files.

Output HDF5 layout (compatible with patcs_artifacts.py):

    data/
      demo_0/
        obs/
          ee_pos      [T, 3]   float32  (observation.state[:3])
          ee_states   [T, 8]   float32  (observation.state)
          joint_state [T, 7]   float32
          image       [T, H, W, 3] uint8
          wrist_image [T, H, W, 3] uint8
        actions       [T, 7]   float32
      demo_1/
        ...

The task name is derived from episode_metadata.file_path (the original HDF5 path
in the LIBERO benchmark). Episodes are grouped by task name, then written one
HDF5 per task under --output/<suite>/<task_name>.hdf5.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def _task_name_from_file_path(file_path: str) -> str:
    """Extract a filesystem-safe task name from the original LIBERO file path.

    Examples
    --------
    .../pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5
        -> pick_up_the_orange_juice_and_place_it_in_the_basket

    .../KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5
        -> KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet
    """
    stem = Path(file_path).stem          # strip directory and extension
    stem = re.sub(r"_demo$", "", stem)   # strip trailing _demo suffix
    return stem


def _iter_episodes(suite_dir: Path, *, skip_images: bool = False):
    """Yield (file_path, arrays) for every episode in a TFDS suite directory.

    Requires tensorflow-datasets (``uv sync --extra rlds``).
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit(
            "tensorflow-datasets is required. Run:\n"
            "  UV_CACHE_DIR=/tmp/uv-cache uv sync --extra rlds"
        ) from exc

    versioned_dirs = sorted(suite_dir.glob("*/"), key=lambda p: p.name)
    if not versioned_dirs:
        raise FileNotFoundError(f"No versioned subdirectory found under {suite_dir}")
    data_dir = versioned_dirs[-1]

    builder = tfds.builder_from_directory(str(data_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)

    for episode in ds:
        file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")

        # steps is a nested tf.data.Dataset; collect all steps into a list first.
        steps_list = list(episode["steps"])

        actions = np.stack([s["action"].numpy() for s in steps_list])                # [T, 7]
        state = np.stack([s["observation"]["state"].numpy() for s in steps_list])    # [T, 8]
        joint_state = np.stack([s["observation"]["joint_state"].numpy() for s in steps_list])  # [T, 7]
        ee_pos = state[:, :3].astype(np.float32)

        arrays: dict = {
            "actions": actions.astype(np.float32),
            "ee_pos": ee_pos,
            "ee_states": state.astype(np.float32),
            "joint_state": joint_state.astype(np.float32),
        }
        if not skip_images:
            arrays["image"] = np.stack(
                [s["observation"]["image"].numpy() for s in steps_list]
            )
            arrays["wrist_image"] = np.stack(
                [s["observation"]["wrist_image"].numpy() for s in steps_list]
            )
        yield file_path, arrays


def convert_suite(
    suite_dir: Path,
    output_dir: Path,
    max_demos_per_task: int | None = None,
    skip_images: bool = False,
) -> dict[str, Path]:
    """Convert one RLDS suite directory to per-task HDF5 files.

    Returns a mapping of task_name -> output HDF5 path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: bucket episodes by task.
    task_episodes: dict[str, list[dict]] = {}
    print(f"Reading {suite_dir.name} ...")
    for file_path, arrays in tqdm(
        _iter_episodes(suite_dir, skip_images=skip_images), desc=suite_dir.name
    ):
        task = _task_name_from_file_path(file_path)
        task_episodes.setdefault(task, []).append(arrays)

    written: dict[str, Path] = {}
    for task, episodes in sorted(task_episodes.items()):
        if max_demos_per_task is not None:
            episodes = episodes[:max_demos_per_task]
        out_path = output_dir / f"{task}.hdf5"
        with h5py.File(out_path, "w") as hf:
            data_grp = hf.create_group("data")
            for i, ep in enumerate(episodes):
                demo_grp = data_grp.create_group(f"demo_{i}")
                obs_grp = demo_grp.create_group("obs")
                obs_grp.create_dataset("ee_pos", data=ep["ee_pos"], compression="gzip")
                obs_grp.create_dataset("ee_states", data=ep["ee_states"], compression="gzip")
                obs_grp.create_dataset("joint_state", data=ep["joint_state"], compression="gzip")
                if not skip_images:
                    obs_grp.create_dataset(
                        "image", data=ep["image"], compression="gzip",
                        chunks=(1, *ep["image"].shape[1:]),
                    )
                    obs_grp.create_dataset(
                        "wrist_image", data=ep["wrist_image"], compression="gzip",
                        chunks=(1, *ep["wrist_image"].shape[1:]),
                    )
                demo_grp.create_dataset("actions", data=ep["actions"], compression="gzip")
            hf.attrs["num_demos"] = len(episodes)
            hf.attrs["task_name"] = task
        print(f"  {task}: {len(episodes)} demos -> {out_path}")
        written[task] = out_path

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OpenVLA modified LIBERO RLDS (tfrecord) to per-task HDF5 files "
            "compatible with new-il-build-patcs-artifacts."
        )
    )
    parser.add_argument(
        "--rlds-root",
        type=Path,
        default=Path("data/openvla_modified_libero_rlds"),
        help="Root directory containing suite subdirectories (libero_object_no_noops, ...).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/libero_rlds_hdf5"),
        help="Output root; one subdirectory per suite will be created here.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["object", "spatial", "goal", "long"],
        choices=["object", "spatial", "goal", "long"],
        help="Which suites to convert (default: all four).",
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=None,
        help="Cap demos per task (useful for smoke testing).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip writing image arrays (much smaller files; not needed for PATCS artifacts).",
    )
    args = parser.parse_args()

    suite_dir_names = {
        "object": "libero_object_no_noops",
        "spatial": "libero_spatial_no_noops",
        "goal": "libero_goal_no_noops",
        "long": "libero_10_no_noops",
    }

    for suite in args.suites:
        suite_dir = args.rlds_root / suite_dir_names[suite]
        if not suite_dir.exists():
            print(f"WARNING: {suite_dir} not found, skipping.")
            continue
        out_suite = args.output / suite
        convert_suite(
            suite_dir,
            out_suite,
            max_demos_per_task=args.max_demos_per_task,
            skip_images=args.skip_images,
        )

    print(f"\nDone. HDF5 files written under {args.output}")
    print("Next step:")
    print(
        "  for each suite/<task>.hdf5 run:\n"
        "  UV_CACHE_DIR=/tmp/uv-cache uv run new-il-build-patcs-artifacts "
        "--input <task>.hdf5 --output data/patcs_artifacts/<suite>/<task>_patcs.npz "
        "--num-demos 16 --num-phase 64 --obs-key ee_pos"
    )
