import importlib
import sys
from pathlib import Path

import pytest


def test_project_libero_symlink_points_to_checkout() -> None:
    root = Path("LIBERO")

    assert root.is_symlink()
    assert root.resolve() == Path("/data/L202500340/Mem/third_party/LIBERO")
    assert (root / "libero" / "libero").is_dir()
    assert (root / "benchmark_scripts" / "download_libero_datasets.py").is_file()


def test_libero_python_package_importable_from_project_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = Path("LIBERO")
    monkeypatch.syspath_prepend(str(root.resolve()))
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_config"))
    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            sys.modules.pop(module_name)

    libero_api = pytest.importorskip("libero.libero")
    benchmark = importlib.import_module("libero.libero.benchmark")

    assert hasattr(libero_api, "get_libero_path")
    assert hasattr(benchmark, "get_benchmark_dict")
