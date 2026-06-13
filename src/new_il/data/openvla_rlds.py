from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


OPENVLA_RLDS_REPO = "openvla/modified_libero_rlds"
OPENVLA_RLDS_REVISION = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"

SUITES = {
    "spatial": "libero_spatial_no_noops",
    "object": "libero_object_no_noops",
    "goal": "libero_goal_no_noops",
    "long": "libero_10_no_noops",
    "10": "libero_10_no_noops",
}


def _suite_names(raw: list[str]) -> list[str]:
    if not raw or raw == ["all"]:
        return ["spatial", "object", "goal", "long"]
    normalized = []
    for item in raw:
        key = item.lower()
        if key not in SUITES:
            raise SystemExit(f"Unknown suite '{item}'. Choose from: all, {', '.join(SUITES)}")
        normalized.append("long" if key == "10" else key)
    return list(dict.fromkeys(normalized))


def _dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def write_manifest(output: Path, suites: list[str], revision: str) -> Path:
    manifest = {
        "repo_id": OPENVLA_RLDS_REPO,
        "revision": revision,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suites": [],
    }
    for suite in suites:
        directory = output / SUITES[suite]
        manifest["suites"].append(
            {
                "suite": suite,
                "directory": str(directory),
                "exists": directory.exists(),
                "size_bytes": _dir_size(directory) if directory.exists() else 0,
            }
        )
    manifest_path = output / "new_il_openvla_rlds_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OpenVLA's filtered modified LIBERO RLDS datasets."
    )
    parser.add_argument("--output", type=Path, default=Path("data/openvla_modified_libero_rlds"))
    parser.add_argument("--revision", default=OPENVLA_RLDS_REVISION)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["all"],
        help="Subset to download: all, spatial, object, goal, long/10.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    suites = _suite_names(args.suites)
    allow_patterns = ["README.md", ".gitattributes"]
    allow_patterns.extend(f"{SUITES[suite]}/**" for suite in suites)
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. Run: UV_CACHE_DIR=/tmp/uv-cache uv sync --extra data"
        ) from exc

    snapshot_download(
        repo_id=OPENVLA_RLDS_REPO,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
        local_files_only=args.local_files_only,
    )
    manifest_path = write_manifest(args.output, suites, args.revision)
    print(f"Wrote {manifest_path}")
