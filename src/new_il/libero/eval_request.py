from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a New-IL LIBERO eval request to a running server.")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    request = Request(
        args.server_url.rstrip("/") + "/eval",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout_sec) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            result = {
                "status": "failed",
                "reason": f"LIBERO eval server HTTP {exc.code}: {body}",
                "server_url": args.server_url,
            }
    except (OSError, URLError, TimeoutError) as exc:
        result = {
            "status": "failed",
            "reason": f"LIBERO eval server request failed: {type(exc).__name__}: {exc}",
            "server_url": args.server_url,
        }

    _write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
