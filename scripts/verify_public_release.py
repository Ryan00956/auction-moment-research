from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_EXTENSIONS = {
    ".dll",
    ".docx",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".onnx",
    ".pcap",
    ".png",
    ".pt",
    ".webp",
    ".xlsx",
    ".ys",
}
RAW_SESSION_ID = re.compile(r"\b\d{8}T\d{6}\b")
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\\?\\)")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PUBLIC_ARTIFACT_ROOTS = ("data", "reports", "results")
CATALOG_PREVIEW_ROOT = (
    REPOSITORY_ROOT
    / "apps"
    / "ephemeral-assistant"
    / "src"
    / "auction_moment_assistant"
    / "catalog_previews"
)
CATALOG_PREVIEW_NAME = re.compile(r"C\d{3}\.png")
IGNORED_SCAN_DIRECTORIES = {
    ".git",
    ".venv",
    "build",
    "dist",
    "__pycache__",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data_manifest(errors: list[str]) -> None:
    manifest_path = REPOSITORY_ROOT / "data" / "v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path, expected in manifest["files"].items():
        path = REPOSITORY_ROOT / "data" / "v1" / relative_path
        if not path.is_file():
            errors.append(f"missing manifest file: {relative_path}")
            continue
        if sha256_file(path) != expected["sha256"]:
            errors.append(f"sha256 mismatch: data/v1/{relative_path}")
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"byte count mismatch: data/v1/{relative_path}")
    for relative_path, expected in manifest.get("reports", {}).items():
        path = REPOSITORY_ROOT / "reports" / relative_path
        if not path.is_file():
            errors.append(f"missing report: {relative_path}")
            continue
        if sha256_file(path) != expected["sha256"]:
            errors.append(f"report sha256 mismatch: {relative_path}")


def verify_result_manifests(errors: list[str]) -> None:
    for manifest_path in (REPOSITORY_ROOT / "results").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for expected in manifest.get("outputs", []):
            path = manifest_path.parent / expected["path"]
            if not path.is_file():
                errors.append(f"missing result: {path.relative_to(REPOSITORY_ROOT)}")
                continue
            if sha256_file(path) != expected["sha256"]:
                errors.append(f"result sha256 mismatch: {path.relative_to(REPOSITORY_ROOT)}")
            if path.stat().st_size != expected["bytes"]:
                errors.append(f"result byte count mismatch: {path.relative_to(REPOSITORY_ROOT)}")


def verify_repository_files(errors: list[str]) -> None:
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or any(
            part in IGNORED_SCAN_DIRECTORIES for part in path.parts
        ):
            continue
        allowed_catalog_preview = (
            path.parent == CATALOG_PREVIEW_ROOT
            and CATALOG_PREVIEW_NAME.fullmatch(path.name) is not None
        )
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS and not allowed_catalog_preview:
            errors.append(f"forbidden extension: {path.relative_to(REPOSITORY_ROOT)}")
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"file exceeds 50 MiB: {path.relative_to(REPOSITORY_ROOT)}")


def verify_catalog_previews(errors: list[str]) -> None:
    metadata_path = CATALOG_PREVIEW_ROOT.parent / "catalog-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {f"{catalog_id}.png" for catalog_id in metadata.get("names") or {}}
    observed = {path.name for path in CATALOG_PREVIEW_ROOT.glob("*.png")}
    if len(expected) != 120:
        errors.append(f"catalog metadata expected 120 names, found {len(expected)}")
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"catalog previews missing: {missing}")
    if extra:
        errors.append(f"catalog previews unexpected: {extra}")
    for path in CATALOG_PREVIEW_ROOT.glob("*.png"):
        header = path.read_bytes()[:24]
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append(f"invalid catalog preview PNG: {path.name}")
            continue
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        if (width, height) != (80, 80):
            errors.append(
                f"catalog preview must be 80x80: {path.name} is {width}x{height}"
            )


def verify_assistant_privacy_boundary(errors: list[str]) -> None:
    source_root = (
        REPOSITORY_ROOT
        / "apps"
        / "ephemeral-assistant"
        / "src"
        / "auction_moment_assistant"
    )
    forbidden_tokens = {
        "auction_protocol": "protocol implementation",
        "packet_realtime_bidder": "packet bidder",
        "tcpdump": "packet capture",
        "write_text(": "runtime text persistence",
        "write_bytes(": "runtime binary persistence",
        "cv2.imwrite(": "runtime image persistence",
        "image.save(": "runtime image persistence",
        "filehandler(": "file logging",
    }
    for path in source_root.glob("*.py"):
        lowered = path.read_text(encoding="utf-8").lower()
        for token, label in forbidden_tokens.items():
            if token in lowered:
                errors.append(
                    f"assistant privacy boundary contains {label}: "
                    f"{path.relative_to(REPOSITORY_ROOT)}"
                )


def verify_public_artifacts(errors: list[str]) -> None:
    for root_name in PUBLIC_ARTIFACT_ROOTS:
        root = REPOSITORY_ROOT / root_name
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"non-UTF-8 artifact: {path.relative_to(REPOSITORY_ROOT)}")
                continue
            if RAW_SESSION_ID.search(text):
                errors.append(f"raw session id found: {path.relative_to(REPOSITORY_ROOT)}")
            if ABSOLUTE_WINDOWS_PATH.search(text):
                errors.append(f"absolute Windows path found: {path.relative_to(REPOSITORY_ROOT)}")
            if EMAIL.search(text):
                errors.append(f"email found: {path.relative_to(REPOSITORY_ROOT)}")

    opponent_header = (REPOSITORY_ROOT / "data" / "v1" / "core" / "opponent_bids.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0].split(",")
    forbidden_columns = {"player_name", "name", "source_file", "captured_at", "created_at"}
    leaked = sorted(forbidden_columns.intersection(opponent_header))
    if leaked:
        errors.append(f"private opponent columns found: {leaked}")


def main() -> int:
    errors: list[str] = []
    verify_repository_files(errors)
    verify_catalog_previews(errors)
    verify_data_manifest(errors)
    verify_result_manifests(errors)
    verify_public_artifacts(errors)
    verify_assistant_privacy_boundary(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    file_count = sum(1 for path in REPOSITORY_ROOT.rglob("*") if path.is_file())
    print(json.dumps({"status": "ok", "files_checked": file_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
