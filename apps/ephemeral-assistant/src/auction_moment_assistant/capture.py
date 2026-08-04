from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from PIL import Image


EXPECTED_SCREEN_SIZE = (1280, 720)


class CaptureError(RuntimeError):
    pass


class FrameSource(Protocol):
    def capture(self) -> np.ndarray: ...


def run_adb(
    adb: str,
    serial: str,
    args: Iterable[str],
    *,
    binary: bool = False,
    timeout: int = 20,
) -> bytes | str:
    command = [str(adb)]
    if serial:
        command.extend(["-s", str(serial)])
    command.extend(str(value) for value in args)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureError(f"ADB 调用失败：{exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise CaptureError(f"ADB 命令失败：{stderr}")
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8", errors="replace")


def discover_serial(adb: str, requested: str = "") -> str:
    output = str(run_adb(adb, "", ["devices"]))
    devices = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    if requested:
        if requested not in devices:
            raise CaptureError(
                f"指定设备 {requested!r} 不在线；当前设备：{devices or '无'}"
            )
        return requested
    if len(devices) != 1:
        raise CaptureError(
            f"需要且只能有一个 ADB 设备；当前设备：{devices or '无'}"
        )
    return devices[0]


@dataclass
class AdbFrameSource:
    adb: str = "adb"
    serial: str = ""
    expected_size: tuple[int, int] = EXPECTED_SCREEN_SIZE

    def connect(self) -> str:
        self.serial = discover_serial(self.adb, self.serial)
        return self.serial

    def capture(self) -> np.ndarray:
        if not self.serial:
            self.connect()
        payload = run_adb(
            self.adb,
            self.serial,
            ["exec-out", "screencap", "-p"],
            binary=True,
            timeout=30,
        )
        assert isinstance(payload, bytes)
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise CaptureError("ADB 返回的截图不是有效 PNG")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                image = source.convert("RGB")
                image.load()
        except OSError as exc:
            raise CaptureError(f"无法解码 ADB 截图：{exc}") from exc
        if image.size != self.expected_size:
            raise CaptureError(
                f"截图尺寸为 {image.size}，当前只校准到 {self.expected_size}"
            )
        return np.asarray(image, dtype=np.uint8).copy()


@dataclass
class StaticFrameSource:
    frame: np.ndarray

    def capture(self) -> np.ndarray:
        return np.asarray(self.frame, dtype=np.uint8).copy()
