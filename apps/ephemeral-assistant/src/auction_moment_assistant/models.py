from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping


MODEL_RELEASE_SCHEMA = "auction-vision-model-release-v1"


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogItem:
    catalog_id: str
    name: str
    quality: str
    width: int
    height: int
    value: int

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_manifest(path: Path | None = None) -> dict:
    if path is None:
        resource = resources.files(__package__).joinpath("model-release.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != MODEL_RELEASE_SCHEMA:
        raise ModelError("模型 Release 清单版本不兼容")
    return payload


def verify_model_directory(
    directory: Path,
    manifest: Mapping | None = None,
) -> dict[str, Path]:
    directory = Path(directory).resolve()
    payload = dict(manifest or load_release_manifest())
    resolved = {}
    errors = []
    for record in payload.get("models") or []:
        filename = str(record["filename"])
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            errors.append(f"非法模型文件名：{filename!r}")
            continue
        path = directory / filename
        if not path.is_file():
            errors.append(f"缺少 {filename}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"{filename} 文件大小不匹配")
            continue
        if sha256_file(path) != str(record["sha256"]):
            errors.append(f"{filename} SHA-256 不匹配")
            continue
        resolved[str(record["role"])] = path
    if errors:
        raise ModelError("；".join(errors))
    return resolved


def download_models(
    directory: Path,
    *,
    base_url: str | None,
    manifest: Mapping | None = None,
) -> dict[str, Path]:
    payload = dict(manifest or load_release_manifest())
    resolved_base = str(base_url or payload.get("download_base_url") or "").rstrip("/")
    if not resolved_base:
        raise ModelError("模型清单尚未配置下载地址，请传入 --model-base-url")
    if not resolved_base.lower().startswith("https://"):
        raise ModelError("模型下载地址必须使用 HTTPS")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for record in payload.get("models") or []:
        filename = str(record["filename"])
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ModelError(f"非法模型文件名：{filename!r}")
        target = directory / filename
        if (
            target.is_file()
            and target.stat().st_size == int(record["bytes"])
            and sha256_file(target) == str(record["sha256"])
        ):
            continue
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".download", dir=directory
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            urllib.request.urlretrieve(
                f"{resolved_base}/{filename}", temporary
            )
            if temporary.stat().st_size != int(record["bytes"]):
                raise ModelError(f"{filename} 下载大小不匹配")
            if sha256_file(temporary) != str(record["sha256"]):
                raise ModelError(f"{filename} 下载 SHA-256 不匹配")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return verify_model_directory(directory, payload)


def load_catalog(treasures_csv: Path) -> tuple[CatalogItem, ...]:
    catalog: dict[str, CatalogItem] = {}
    with Path(treasures_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            catalog_id = str(row.get("catalog_id") or "")
            if not catalog_id:
                continue
            item = CatalogItem(
                catalog_id=catalog_id,
                name=f"catalog_{catalog_id}",
                quality=str(row["quality"]),
                width=int(float(row["width"])),
                height=int(float(row["height"])),
                value=int(float(row["value"])),
            )
            previous = catalog.get(catalog_id)
            if previous is not None and previous != item:
                raise ModelError(f"公开图鉴存在冲突：{catalog_id}")
            catalog[catalog_id] = item
    if not catalog:
        raise ModelError("公开图鉴为空")
    return tuple(catalog[key] for key in sorted(catalog))
