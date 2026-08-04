from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SCHEMA_VERSION = "auction-vision-model-release-v1"
WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?![\\/])|\\\\\?\\)"
)
RAW_SESSION_ID = re.compile(r"\b\d{8}T\d{6}\b")
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    role: str
    expected_task: str
    expected_classes: int


MODEL_SPECS = (
    ModelSpec(
        "clue-spatial-yolo26n-v1.pt",
        "spatial_detector",
        "detect",
        3,
    ),
    ModelSpec(
        "clue-detector-yolo26n-v1.pt",
        "attribute_detector",
        "detect",
        13,
    ),
    ModelSpec(
        "treasure-classifier-yolo26n-v1.pt",
        "identity_classifier",
        "classify",
        120,
    ),
)


class ReleaseBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_strings(value: Any, path: str = "root", depth: int = 0):
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_strings(child, f"{path}.{key}", depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}[{index}]", depth + 1)
    elif isinstance(value, (str, Path)):
        yield path, str(value)


def suspicious_metadata(value: Any) -> list[dict[str, str]]:
    findings = []
    for path, text in _walk_strings(value):
        if WINDOWS_PATH.search(text):
            findings.append(
                {"path": path, "value": text, "reason": "windows_path"}
            )
        elif RAW_SESSION_ID.search(text):
            findings.append(
                {"path": path, "value": text, "reason": "raw_session_id"}
            )
    return findings


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _safe_train_args(task: str, filename: str, imgsz: Any) -> dict:
    return {
        "task": str(task),
        "model": filename,
        "imgsz": imgsz if isinstance(imgsz, (int, list, tuple)) else 640,
        "single_cls": False,
    }


def sanitize_model(source: Path, target: Path, spec: ModelSpec) -> dict:
    try:
        from ultralytics import YOLO, __version__ as ultralytics_version
    except ImportError as exc:
        raise ReleaseBuildError(
            "模型净化需要安装与训练时兼容的 ultralytics"
        ) from exc

    loaded = YOLO(str(source.resolve()))
    if str(loaded.task) != spec.expected_task:
        raise ReleaseBuildError(
            f"{source.name} task={loaded.task!r}，期望 {spec.expected_task!r}"
        )
    names = dict(getattr(loaded.model, "names", {}) or {})
    if len(names) != spec.expected_classes:
        raise ReleaseBuildError(
            f"{source.name} 类别数 {len(names)}，期望 {spec.expected_classes}"
        )

    before_state = state_sha256(loaded.model)
    raw_checkpoint_findings = suspicious_metadata(loaded.ckpt or {})
    original_args = dict(getattr(loaded.model, "args", {}) or {})
    safe_args = _safe_train_args(
        str(loaded.task),
        spec.filename,
        original_args.get("imgsz", 640),
    )
    loaded.model.args = dict(safe_args)
    retained_checkpoint = {
        key: value
        for key, value in (loaded.ckpt or {}).items()
        if key
        not in {
            "model",
            "ema",
            "optimizer",
            "scaler",
            "git",
            "train_args",
        }
    }
    retained_checkpoint["train_args"] = dict(safe_args)
    loaded.ckpt = retained_checkpoint
    loaded.save(str(target))

    reloaded = YOLO(str(target.resolve()))
    after_state = state_sha256(reloaded.model)
    if after_state != before_state:
        raise ReleaseBuildError(f"{source.name} 净化后参数哈希发生变化")
    try:
        import torch

        raw_checkpoint = torch.load(
            target,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ReleaseBuildError(
            f"无法复核净化后的检查点 {target.name}: {exc}"
        ) from exc
    remaining_findings = suspicious_metadata(raw_checkpoint)
    if remaining_findings:
        raise ReleaseBuildError(
            f"{source.name} 净化后仍含敏感元数据：{remaining_findings}"
        )
    return {
        "filename": target.name,
        "role": spec.role,
        "format": "ultralytics-pytorch",
        "task": str(reloaded.task),
        "class_count": len(names),
        "class_names": [str(names[index]) for index in sorted(names)],
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "state_sha256": after_state,
        "source_metadata_findings_removed": len(raw_checkpoint_findings),
        "ultralytics_version": str(ultralytics_version),
        "license": "AGPL-3.0",
    }


def verify_public_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"无法读取世界模型 {path}: {exc}") from exc
    findings = suspicious_metadata(payload)
    if findings:
        raise ReleaseBuildError(f"世界模型包含私有元数据：{findings}")
    return payload


def locate_ultralytics_license() -> Path:
    distribution = importlib.metadata.distribution("ultralytics")
    candidates = [
        distribution.locate_file(file)
        for file in distribution.files or ()
        if "license" in str(file).lower()
        and Path(str(file)).suffix.lower() not in {".py", ".pyc"}
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and "GNU AFFERO GENERAL PUBLIC LICENSE" in path.read_text(
            encoding="utf-8", errors="ignore"
        ):
            return path
    raise ReleaseBuildError("未找到 Ultralytics 随附的 AGPL-3.0 许可证文本")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def deterministic_zip(target: Path, files: Iterable[Path], root: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build_release(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise ReleaseBuildError(f"输出目录已经存在，拒绝覆盖：{output_root}")
    output_root.mkdir(parents=True)

    model_records = []
    for spec in MODEL_SPECS:
        source = source_root / spec.filename
        if not source.is_file():
            raise ReleaseBuildError(f"缺少模型：{source}")
        model_records.append(
            sanitize_model(source, output_root / spec.filename, spec)
        )

    world_model = args.world_model.resolve()
    world_payload = verify_public_json(world_model)
    world_target = output_root / "world-model-v1-S.json"
    write_json(world_target, world_payload)
    model_records.append(
        {
            "filename": world_target.name,
            "role": "probabilistic_world_model",
            "format": "json",
            "schema_version": world_payload.get("schema_version"),
            "bytes": world_target.stat().st_size,
            "sha256": sha256_file(world_target),
            "license": "Apache-2.0",
        }
    )

    shutil.copy2(args.model_card.resolve(), output_root / "MODEL_CARD.md")
    shutil.copy2(
        args.notices.resolve(), output_root / "THIRD_PARTY_NOTICES.md"
    )
    shutil.copy2(locate_ultralytics_license(), output_root / "AGPL-3.0.txt")

    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release": str(args.release),
        "privacy_contract": {
            "contains_raw_images": False,
            "contains_session_records": False,
            "contains_packet_capture_or_protocol_code": False,
            "contains_absolute_training_paths": False,
        },
        "models": model_records,
    }
    manifest_path = output_root / "model-manifest.json"
    write_json(manifest_path, manifest)

    checksum_lines = []
    release_files = [path for path in output_root.iterdir() if path.is_file()]
    for path in sorted(release_files, key=lambda item: item.name):
        checksum_lines.append(f"{sha256_file(path)}  {path.name}")
    checksum_path = output_root / "SHA256SUMS"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    archive_path = output_root.parent / f"auction-assistant-{args.release}.zip"
    deterministic_zip(
        archive_path,
        [path for path in output_root.iterdir() if path.is_file()],
        output_root,
    )
    return {
        "status": "ok",
        "release_directory": str(output_root),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "models": model_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="净化中间态 YOLO 权重并构建可公开的 GitHub Release 资产"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--world-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", default="vision-models-v1.0.0")
    parser.add_argument(
        "--model-card",
        type=Path,
        default=REPOSITORY_ROOT / "models" / "VISION_MODEL_CARD.md",
    )
    parser.add_argument(
        "--notices",
        type=Path,
        default=REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md",
    )
    return parser


def main() -> int:
    try:
        result = build_release(build_parser().parse_args())
    except ReleaseBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
