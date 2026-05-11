from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def _count_demos(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        if "data" in handle:
            return len(handle["data"].keys())
        return sum(1 for key in handle.keys() if key.startswith("demo"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small reproducible LIBERO HDF5 manifest.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--max-demos-per-file", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/libero_smoke_manifest.json"))
    args = parser.parse_args()

    files = sorted((args.dataset_root / args.suite).glob("*.hdf5"))[: args.max_files]
    if not files:
        raise SystemExit(f"No .hdf5 files found under {args.dataset_root / args.suite}")

    manifest = {
        "suite": args.suite,
        "dataset_root": str(args.dataset_root),
        "max_demos_per_file": args.max_demos_per_file,
        "files": [
            {
                "path": str(path),
                "available_demos": _count_demos(path),
                "selected_demos": min(_count_demos(path), args.max_demos_per_file),
            }
            for path in files
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
