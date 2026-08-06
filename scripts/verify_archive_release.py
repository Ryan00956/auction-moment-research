from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ARCHIVE_ID = "season-2026-archive-v1"
RELEASE_MANIFEST = "archive-release-manifest.json"
CHECKSUMS = "SHA256SUMS.txt"
REQUIRED_FILES = {
    "auction-moment-public-archive-v1.zip",
    "auction_moment_research-0.1.0-py3-none-any.whl",
    "auction_moment_assistant-0.2.0b6-py3-none-any.whl",
    "clue-spatial-yolo26n-v1.pt",
    "clue-detector-yolo26n-v1.pt",
    "treasure-classifier-yolo26n-v1.pt",
    "latest-value-model-v6.joblib",
    "world-model-v2-S.json",
    "LATEST_MODEL_CARD.md",
    "assistant-release-v2.json",
    "STATUS.md",
    "RETURN_RUNBOOK.md",
    "requirements-runtime-win-py312.txt",
    "season-2026-archive-v1.json",
    "RELEASE_NOTES.md",
    "install_archive.ps1",
    RELEASE_MANIFEST,
    CHECKSUMS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(bundle_directory: Path) -> list[str]:
    errors: list[str] = []
    bundle_directory = Path(bundle_directory)
    observed = {path.name for path in bundle_directory.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - observed)
    if missing:
        errors.append(f"missing required files: {missing}")
        return errors

    checksum_path = bundle_directory / CHECKSUMS
    declared_checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if not separator or not filename:
            errors.append(f"malformed checksum line: {line!r}")
            continue
        declared_checksums[filename] = digest
    for filename, expected in declared_checksums.items():
        path = bundle_directory / filename
        if not path.is_file():
            errors.append(f"checksummed file missing: {filename}")
        elif sha256_file(path) != expected:
            errors.append(f"checksum mismatch: {filename}")

    manifest = json.loads(
        (bundle_directory / RELEASE_MANIFEST).read_text(encoding="utf-8")
    )
    if manifest.get("archive_id") != ARCHIVE_ID:
        errors.append("archive id mismatch")
    if manifest.get("project_status") != "hibernating":
        errors.append("project status must be hibernating")
    boundary = manifest.get("capability_boundary") or {}
    for field in (
        "formal_full_match_end_to_end_completed",
        "fresh_prospective_validation_completed",
        "production_enabled",
        "automatic_bid_execution_in_public_project",
        "all_assistant_outputs_actionable",
    ):
        if boundary.get(field) is not False:
            errors.append(f"capability boundary must keep {field}=false")

    for payload in manifest.get("payloads") or []:
        filename = payload.get("filename")
        path = bundle_directory / str(filename)
        if not path.is_file():
            errors.append(f"manifest payload missing: {filename}")
            continue
        if path.stat().st_size != payload.get("bytes"):
            errors.append(f"manifest byte count mismatch: {filename}")
        if sha256_file(path) != payload.get("sha256"):
            errors.append(f"manifest sha256 mismatch: {filename}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a canonical public hibernation Release directory."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify_bundle(args.bundle_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    manifest = json.loads(
        (args.bundle_dir / RELEASE_MANIFEST).read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "archive_id": manifest["archive_id"],
                "source_commit": manifest["source_commit"],
                "payloads": len(manifest["payloads"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
