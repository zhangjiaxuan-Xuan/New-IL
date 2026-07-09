from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


LIBERO_DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


@dataclass(frozen=True)
class OpenPILiberoConfig:
    """Observation contract used by upstream OpenPI's LIBERO pi0.5 example."""

    resize_size: int = 224
    agent_image_key: str = "agentview_image"
    wrist_image_key: str = "robot0_eye_in_hand_image"


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)[:4].copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3])) / den)).astype(np.float32, copy=False)


def libero_state_for_openpi(obs: dict[str, Any]) -> np.ndarray:
    """Build OpenPI's 8D LIBERO state: eef xyz, axis-angle orientation, gripper qpos."""

    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3],
            quat_to_axis_angle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2],
        )
    ).astype(np.float32, copy=False)


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {image.shape}")
    return np.ascontiguousarray(image)


def resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    """Resize one HWC image to square size without distorting aspect ratio."""

    from PIL import Image

    image = to_uint8_image(image)
    height, width = image.shape[:2]
    if height == size and width == size:
        return image
    ratio = max(width / size, height / size)
    resized_width = max(1, int(width / ratio))
    resized_height = max(1, int(height / ratio))
    resized = Image.fromarray(image).resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), 0)
    canvas.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def openpi_libero_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Match OpenPI LIBERO preprocessing: rotate 180 degrees, then resize/pad."""

    rotated = np.ascontiguousarray(to_uint8_image(image)[::-1, ::-1])
    return resize_with_pad(rotated, size)


def build_openpi_libero_payload(
    obs: dict[str, Any],
    prompt: str,
    config: OpenPILiberoConfig = OpenPILiberoConfig(),
) -> dict[str, Any]:
    """Convert a LIBERO observation dict to OpenPI's websocket policy payload."""

    return {
        "observation/image": openpi_libero_image(obs[config.agent_image_key], config.resize_size),
        "observation/wrist_image": openpi_libero_image(obs[config.wrist_image_key], config.resize_size),
        "observation/state": libero_state_for_openpi(obs),
        "prompt": prompt,
    }


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class OpenPIWebsocketPolicy:
    """Minimal client for OpenPI's websocket policy server.

    This intentionally mirrors `openpi-client` without importing OpenPI, so the
    New-IL Python 3.10 rollout process can talk to a Python 3.11 OpenPI server.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str | None = None,
        retry_interval_sec: float = 5.0,
    ) -> None:
        import msgpack
        import websockets.sync.client

        self._msgpack = msgpack
        uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None and ":" not in uri.rsplit("/", 1)[-1]:
            uri = f"{uri}:{port}"
        self.uri = uri
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    additional_headers=headers,
                )
                metadata = self._ws.recv()
                if isinstance(metadata, str):
                    raise RuntimeError(f"OpenPI server returned text metadata: {metadata}")
                self.metadata = self._msgpack.unpackb(metadata, object_hook=_unpack_array)
                break
            except ConnectionRefusedError:
                time.sleep(retry_interval_sec)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._msgpack.packb(payload, default=_pack_array))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server returned an error:\n{response}")
        return self._msgpack.unpackb(response, object_hook=_unpack_array)

    def action_chunk(self, payload: dict[str, Any]) -> np.ndarray:
        response = self.infer(payload)
        if "actions" not in response:
            raise KeyError(f"OpenPI response missing 'actions'; keys={sorted(response)}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != 7:
            raise ValueError(f"expected OpenPI actions [H, 7], got {actions.shape}")
        return actions
