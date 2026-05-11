from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from new_il.libero.evaluate_checkpoint import evaluate
from new_il.libero.evaluate_checkpoint import _import_libero_api


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _prepare_paths(libero_root: Path | None) -> None:
    if libero_root is None:
        return
    libero_root = libero_root.expanduser().resolve()
    package_root = libero_root / "libero"
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    os.environ.setdefault("LIBERO_ROOT", str(libero_root))
    config_dir = Path(os.environ.get("NEW_IL_LIBERO_CONFIG_PATH", "/tmp/new_il_libero_config"))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    if not config_path.exists():
        benchmark_root = package_root / "libero"
        config_path.write_text(
            "\n".join(
                [
                    f"benchmark_root: {benchmark_root}",
                    f"bddl_files: {benchmark_root / 'bddl_files'}",
                    f"init_states: {benchmark_root / 'init_files'}",
                    f"datasets: {package_root / 'datasets'}",
                    f"assets: {benchmark_root / 'assets'}",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _import_status() -> dict[str, Any]:
    status: dict[str, Any] = {}
    for name in ("libero", "robosuite", "imageio", "torch"):
        try:
            module = __import__(name)
            status[name] = {"ok": True, "file": getattr(module, "__file__", None)}
        except Exception as exc:
            status[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return status


def _benchmark_tasks(benchmark_name: str) -> dict[str, Any]:
    try:
        benchmark, _, _ = _import_libero_api()
        names = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] if benchmark_name == "all" else [benchmark_name]
        tasks = []
        for name in names:
            task_suite = benchmark.get_benchmark_dict()[name]()
            for idx in range(task_suite.n_tasks):
                task = task_suite.get_task(idx)
                tasks.append(
                    {
                        "benchmark": name,
                        "task_id": idx,
                        "name": getattr(task, "name", None),
                        "language": getattr(task, "language", None),
                        "problem_folder": getattr(task, "problem_folder", None),
                        "bddl_file": getattr(task, "bddl_file", None),
                    }
                )
        return {"status": "completed", "benchmark": benchmark_name, "tasks": tasks}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _namespace_from_payload(payload: dict[str, Any], defaults: argparse.Namespace) -> argparse.Namespace:
    merged = vars(defaults).copy()
    merged.update(payload)
    for key in ("checkpoint", "output_dir", "libero_root"):
        if merged.get(key) is not None:
            merged[key] = Path(merged[key])
    return argparse.Namespace(**merged)


class LiberoEvalHandler(BaseHTTPRequestHandler):
    server_version = "NewILLiberoEval/0.1"

    def log_message(self, fmt: str, *args) -> None:
        if self.server.verbose:
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "server": self.server_version,
                    "libero_root": str(self.server.defaults.libero_root)
                    if self.server.defaults.libero_root is not None else None,
                    "imports": _import_status(),
                },
            )
            return
        if parsed.path == "/tasks":
            query = parse_qs(parsed.query)
            benchmark_name = query.get("benchmark", [self.server.defaults.benchmark])[0]
            payload = _benchmark_tasks(benchmark_name)
            _json_response(self, 200 if payload.get("status") == "completed" else 500, payload)
            return
        _json_response(self, 404, {"status": "not_found", "path": parsed.path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/eval":
            _json_response(self, 404, {"status": "not_found", "path": parsed.path})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            eval_args = _namespace_from_payload(payload, self.server.defaults)
            result = evaluate(eval_args)
            status = 200 if result.get("status") in {"completed", "skipped"} else 500
            _json_response(self, status, result)
        except Exception as exc:
            _json_response(
                self,
                500,
                {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LIBERO evaluation requests for New-IL.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--libero-root", type=Path, default=os.environ.get("LIBERO_ROOT"))
    parser.add_argument("--benchmark", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--camera-name", default="agentview_image")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _prepare_paths(args.libero_root)
    server = ThreadingHTTPServer((args.host, args.port), LiberoEvalHandler)
    server.defaults = args
    server.verbose = args.verbose
    print(
        json.dumps(
            {
                "event": "libero_eval_server_start",
                "host": args.host,
                "port": args.port,
                "libero_root": str(args.libero_root) if args.libero_root else None,
                "imports": _import_status(),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
