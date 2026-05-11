from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official LIBERO demonstration datasets.")
    parser.add_argument("--libero-root", type=Path, required=True, help="Local checkout of LIBERO.")
    parser.add_argument(
        "--datasets",
        default="libero_spatial",
        help="Dataset suite accepted by LIBERO, e.g. libero_spatial/libero_object/libero_goal/libero_100.",
    )
    parser.add_argument("--use-huggingface", action="store_true", help="Use LIBERO's Hugging Face mirror.")
    args = parser.parse_args()

    script = args.libero_root / "benchmark_scripts" / "download_libero_datasets.py"
    if not script.exists():
        raise SystemExit(f"Cannot find LIBERO downloader: {script}")

    command = ["python", str(script), "--datasets", args.datasets]
    if args.use_huggingface:
        command.append("--use-huggingface")
    subprocess.run(command, cwd=args.libero_root, check=True)


if __name__ == "__main__":
    main()
