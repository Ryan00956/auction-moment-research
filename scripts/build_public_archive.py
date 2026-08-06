from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ID = "season-2026-archive-v1"
ARCHIVE_DATE = "2026-08-06"
ZIP_TIMESTAMP = (2026, 8, 6, 0, 0, 0)
ARCHIVE_ZIP = "auction-moment-public-archive-v1.zip"
RELEASE_MANIFEST = "archive-release-manifest.json"
CHECKSUMS = "SHA256SUMS.txt"

SNAPSHOT_ROOTS = (
    "data/v1",
    "reports",
    "results",
)
SNAPSHOT_FILES = (
    "README.md",
    "STATUS.md",
    "RETURN_RUNBOOK.md",
    "CITATION.cff",
    "DATA_LICENSE.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "models/LATEST_MODEL_CARD.md",
    "models/MODEL_CARD.md",
    "models/VISION_MODEL_CARD.md",
    "models/assistant-release-v2.json",
    "models/vision-release-v1.json",
    "requirements-lock.txt",
    "requirements-runtime-win-py312.txt",
    "archive/season-2026-archive-v1.json",
    "archive/season-2026-archive-v1.md",
    "scripts/install_archive.ps1",
)
SUPPORT_FILES = {
    "models/LATEST_MODEL_CARD.md": "LATEST_MODEL_CARD.md",
    "STATUS.md": "STATUS.md",
    "RETURN_RUNBOOK.md": "RETURN_RUNBOOK.md",
    "requirements-runtime-win-py312.txt": "requirements-runtime-win-py312.txt",
    "archive/season-2026-archive-v1.json": "season-2026-archive-v1.json",
    "archive/season-2026-archive-v1.md": "RELEASE_NOTES.md",
    "scripts/install_archive.ps1": "install_archive.ps1",
}
WHEEL_REQUIREMENTS = {
    "auction_moment_research-0.1.0-py3-none-any.whl": {
        "bytes": 45548,
        "sha256": "c8c275c4a177995c8d9b5913b28119260a34fbe1dc7aaf47169e0ad10d7d1b76",
    },
    "auction_moment_assistant-0.2.0b6-py3-none-any.whl": {
        "bytes": 1600393,
        "sha256": "26cc89911c3e9008bd0a5db3439bee1c7d0c48592dffb518c4362f2e17353df3",
    },
}
RELEASE_DOCUMENTS = ("assistant-release-v2.json",)


class ArchiveBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def snapshot_entries(root: Path = REPOSITORY_ROOT) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for relative in SNAPSHOT_FILES:
        path = root / relative
        if not path.is_file():
            raise ArchiveBuildError(f"missing snapshot file: {relative}")
        entries.append((path, Path(relative).as_posix()))
    for relative in SNAPSHOT_ROOTS:
        directory = root / relative
        if not directory.is_dir():
            raise ArchiveBuildError(f"missing snapshot directory: {relative}")
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                entries.append((path, path.relative_to(root).as_posix()))
    return sorted(entries, key=lambda entry: entry[1])


def write_deterministic_zip(
    destination: Path,
    entries: Iterable[tuple[Path, str]],
) -> None:
    destination = Path(destination)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source, archive_name in sorted(entries, key=lambda entry: entry[1]):
            info = zipfile.ZipInfo(archive_name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, Path(source).read_bytes(), compresslevel=9)


def _model_requirements() -> list[dict]:
    manifest = json.loads(
        (REPOSITORY_ROOT / "models" / "assistant-release-v2.json").read_text(
            encoding="utf-8"
        )
    )
    return list(manifest["models"])


def _copy_verified_assets(asset_directory: Path, output_directory: Path) -> None:
    asset_directory = Path(asset_directory)
    if not asset_directory.is_dir():
        raise ArchiveBuildError(f"asset directory not found: {asset_directory}")

    required_names = [*WHEEL_REQUIREMENTS, *RELEASE_DOCUMENTS]
    required_names.extend(model["filename"] for model in _model_requirements())
    for name in required_names:
        source = asset_directory / name
        if not source.is_file():
            raise ArchiveBuildError(f"missing release asset: {name}")

    expected_manifest = REPOSITORY_ROOT / "models" / "assistant-release-v2.json"
    downloaded_manifest = asset_directory / "assistant-release-v2.json"
    if downloaded_manifest.read_bytes() != expected_manifest.read_bytes():
        raise ArchiveBuildError("assistant-release-v2.json differs from repository")

    for model in _model_requirements():
        source = asset_directory / model["filename"]
        if source.stat().st_size != model["bytes"]:
            raise ArchiveBuildError(f"model byte count mismatch: {source.name}")
        if sha256_file(source) != model["sha256"]:
            raise ArchiveBuildError(f"model sha256 mismatch: {source.name}")

    for name, expected in WHEEL_REQUIREMENTS.items():
        source = asset_directory / name
        if source.stat().st_size != expected["bytes"]:
            raise ArchiveBuildError(f"wheel byte count mismatch: {source.name}")
        if sha256_file(source) != expected["sha256"]:
            raise ArchiveBuildError(f"wheel sha256 mismatch: {source.name}")

    for name in sorted(set(required_names)):
        shutil.copy2(asset_directory / name, output_directory / name)


def _payload_record(path: Path, role: str) -> dict:
    return {
        "filename": path.name,
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_checksums(output_directory: Path, paths: Sequence[Path]) -> None:
    lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(paths)]
    (output_directory / CHECKSUMS).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_archive(
    asset_directory: Path,
    output_directory: Path,
    *,
    source_ref: str = ARCHIVE_ID,
    allow_dirty: bool = False,
) -> dict:
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ArchiveBuildError(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    if not allow_dirty and _git("status", "--porcelain"):
        raise ArchiveBuildError(
            "repository is dirty; commit before building an archive"
        )

    descriptor = json.loads(
        (REPOSITORY_ROOT / "archive" / "season-2026-archive-v1.json").read_text(
            encoding="utf-8"
        )
    )
    if descriptor.get("archive_id") != ARCHIVE_ID:
        raise ArchiveBuildError("archive descriptor id mismatch")

    archive_zip = output_directory / ARCHIVE_ZIP
    write_deterministic_zip(archive_zip, snapshot_entries())
    _copy_verified_assets(Path(asset_directory), output_directory)
    for source_name, output_name in SUPPORT_FILES.items():
        shutil.copy2(REPOSITORY_ROOT / source_name, output_directory / output_name)

    roles = {
        ARCHIVE_ZIP: "public_data_reports_results_and_cards",
        "auction_moment_research-0.1.0-py3-none-any.whl": "research_wheel",
        "auction_moment_assistant-0.2.0b6-py3-none-any.whl": "assistant_wheel",
        "LATEST_MODEL_CARD.md": "model_card",
        "assistant-release-v2.json": "model_asset_manifest",
        "STATUS.md": "hibernation_status",
        "RETURN_RUNBOOK.md": "return_runbook",
        "requirements-runtime-win-py312.txt": "known_working_runtime_pins",
        "season-2026-archive-v1.json": "archive_contract",
        "RELEASE_NOTES.md": "archive_release_notes",
        "install_archive.ps1": "verified_windows_restore_script",
    }
    for model in _model_requirements():
        roles[model["filename"]] = model["role"]

    payload_paths = sorted(
        path
        for path in output_directory.iterdir()
        if path.is_file() and path.name not in {RELEASE_MANIFEST, CHECKSUMS}
    )
    manifest = {
        "schema_version": "auction-moment-archive-release-v1",
        "archive_id": ARCHIVE_ID,
        "archive_date": ARCHIVE_DATE,
        "source_repository": "Ryan00956/auction-moment-research",
        "source_ref": source_ref,
        "source_commit": _git("rev-parse", "HEAD"),
        "project_status": descriptor["project_status"],
        "event_status": descriptor["event_status"],
        "capability_boundary": descriptor["capability_boundary"],
        "payloads": [
            _payload_record(path, roles.get(path.name, "release_asset"))
            for path in payload_paths
        ],
    }
    manifest_path = output_directory / RELEASE_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_checksums(output_directory, [*payload_paths, manifest_path])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical public hibernation Release assets."
    )
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-ref", default=ARCHIVE_ID)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_archive(
        args.asset_dir,
        args.output_dir,
        source_ref=args.source_ref,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
