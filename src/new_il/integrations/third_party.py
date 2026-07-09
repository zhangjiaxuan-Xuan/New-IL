from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"


@dataclass(frozen=True)
class ThirdPartyCheck:
    name: str
    path: str
    exists: bool
    is_symlink: bool
    target: str | None
    import_name: str | None
    importable: bool | None
    commit: str | None
    notes: str


def _head_commit(path: Path) -> str | None:
    git_dir = path / ".git"
    head = git_dir / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("ref: "):
        return text or None
    ref = text.split(" ", 1)[1]
    ref_path = git_dir / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8", errors="replace").strip() or None
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def _importable(import_name: str, path: Path | None = None) -> bool:
    added: list[str] = []
    if path is not None and path.exists():
        candidates = [path, path / "src", path.parent]
        for candidate in candidates:
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                added.append(str(candidate))
    try:
        return importlib.util.find_spec(import_name) is not None
    finally:
        for item in added:
            try:
                sys.path.remove(item)
            except ValueError:
                pass


def _check(
    name: str,
    rel_path: str,
    import_name: str | None,
    notes: str,
) -> ThirdPartyCheck:
    path = THIRD_PARTY_ROOT / rel_path
    exists = path.exists()
    target = str(path.resolve()) if path.is_symlink() and exists else None
    return ThirdPartyCheck(
        name=name,
        path=str(path),
        exists=exists,
        is_symlink=path.is_symlink(),
        target=target,
        import_name=import_name,
        importable=_importable(import_name, path) if import_name else None,
        commit=_head_commit(path.resolve()) if exists else None,
        notes=notes,
    )


def collect_status() -> dict[str, Any]:
    checks = [
        _check(
            "LIBERO",
            "LIBERO",
            "libero",
            "LIBERO benchmark checkout used for simulation evaluation and task metadata.",
        ),
        _check(
            "robosuite-mem",
            "robosuite",
            "robosuite",
            "Package-name alias to the Mem robosuite fork with EGL device isolation fixes.",
        ),
        _check(
            "openpi",
            "openpi",
            "openpi",
            "OpenPI checkout for pi0/pi0.5 policy serving, fine-tuning, and LIBERO eval.",
        ),
        _check(
            "sdtw-cuda-torch",
            "sdtw-cuda-torch",
            None,
            "Optional differentiable Soft-DTW reference for trajectory-level PATCS alignment.",
        ),
    ]
    return {
        "project_root": str(PROJECT_ROOT),
        "third_party_root": str(THIRD_PARTY_ROOT),
        "checks": [asdict(check) for check in checks],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report New-IL third-party integration status.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    status = collect_status()
    if args.json:
        print(json.dumps(status, indent=2))
        return

    print(f"third_party root: {status['third_party_root']}")
    for check in status["checks"]:
        flag = "OK" if check["exists"] else "MISSING"
        imp = check["importable"]
        imp_text = "n/a" if imp is None else ("importable" if imp else "not importable")
        print(f"- {check['name']}: {flag}, {imp_text}")
        print(f"  path: {check['path']}")
        if check["target"]:
            print(f"  target: {check['target']}")
        if check["commit"]:
            print(f"  commit: {check['commit']}")
        print(f"  note: {check['notes']}")


if __name__ == "__main__":
    main()
